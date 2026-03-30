"""Reflow PDF text for LLM readability: line-wrap hyphens, wrapped lines, noise."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


_PAGE_NUM_LINE = re.compile(
    r"^\s*(?:Page\s*)?\d{1,4}\s*(?:/|\s+of\s+)?\s*\d{0,4}\s*$",
    re.IGNORECASE,
)
_DIGITS_ONLY = re.compile(r"^\s*\d+\s*$")
# Page index artifacts seen in some PDFs (e.g. "    10|Downloaded ...")
_DIGITS_PIPE = re.compile(r"^\s*\d{1,4}\s*\|\s*.*$")
# Margin page index: digits with heavy right-padding (common in two-column PDFs)
_PAGE_NUM_PADDED = re.compile(r"^\s*\d{1,4}\s{8,}$")
_PAGE_HEADER_LINE = re.compile(r"^Page\s+\d{1,4}\s+of\s+\d{1,4}\s*$", re.IGNORECASE)

# Boilerplate removal is driven by config when available.
_DEFAULT_BOILERPLATE_LINE_PATTERNS: list[tuple[str, bool]] = [
    (r"\bDownloaded from\b", True),
    (r"\bCopyright\b", True),
    (r"\ball rights reserved\b", True),
    (r"\bMy\s+Account\b", True),
    (r"Creative Commons", True),
    (r"^my\s+\w+\s+in\s+the\s+journal\s+online\b", True),
]


def _load_boilerplate_line_patterns() -> list[re.Pattern[str]]:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "boilerplate_line_patterns.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            patterns: list[re.Pattern[str]] = []
            for item in data:
                pat = item["pattern"]
                flags = 0
                if item.get("ignore_case", False):
                    flags |= re.IGNORECASE
                patterns.append(re.compile(pat, flags))
            if patterns:
                return patterns
        except Exception:
            pass

    patterns: list[re.Pattern[str]] = []
    for pat, ignore_case in _DEFAULT_BOILERPLATE_LINE_PATTERNS:
        flags = re.IGNORECASE if ignore_case else 0
        patterns.append(re.compile(pat, flags))
    # URLs / DOIs are often boilerplate; skip them too.
    patterns.append(re.compile(r"\bhttps?://\S+", re.IGNORECASE))
    patterns.append(re.compile(r"\bdoi\.org\b", re.IGNORECASE))
    return patterns


_BOILERPLATE_LINE_PATTERNS: list[re.Pattern[str]] = _load_boilerplate_line_patterns()

# Optional leading promo block (some journals)
_LEADING_PROMO_PREFIX = re.compile(
    r"^(?:A Quick Take\s*\n\s*is available at\s*\n\s*NEJM\.org\s*\n?)",
    re.IGNORECASE | re.MULTILINE,
)

# Standalone section headings — do not merge next line into these
_KNOWN_HEADINGS = frozenset(
    {
        "BACKGROUND",
        "METHODS",
        "METHOD",
        "RESULTS",
        "DISCUSSION",
        "CONCLUSIONS",
        "INTRODUCTION",
        "ABSTRACT",
        "REFERENCES",
        "ACKNOWLEDGMENTS",
        "ACKNOWLEDGEMENTS",
        "SUPPLEMENTARY",
        "RESULTS AND DISCUSSION",
        "PATIENTS AND METHODS",
        "MATERIALS AND METHODS",
        "STUDY DESIGN",
        "TRIAL DESIGN",
        "ORIGINAL ARTICLE",
    }
)

# Token fixes can be tweaked without changing code (via `reflow_token_fixes.json`).
_DEFAULT_TOKEN_FIXES: list[tuple[str, str, int]] = [
    # Generic OCR/PDF extraction glitch
    (r"\bABSTR\s+ACT\b", "ABSTRACT", re.IGNORECASE),
    # Publisher header fragments (optional; can be edited/extended by user)
    (r"\bThe\s+new\s+engl\s+and\s+jour\s+nal\s+of\s+medicine\b", "The New England Journal of Medicine", re.IGNORECASE),
    (r"\bn\s+engl\s+j\s+med\b", "N Engl J Med", re.IGNORECASE),
]


def _load_token_fixes() -> list[tuple[re.Pattern[str], str]]:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "reflow_token_fixes.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            fixes: list[tuple[re.Pattern[str], str]] = []
            for item in data:
                pattern = item["pattern"]
                replacement = item["replacement"]
                flags = 0
                if item.get("ignore_case", False):
                    flags |= re.IGNORECASE
                fixes.append((re.compile(pattern, flags), replacement))
            if fixes:
                return fixes
        except Exception:
            # If a user config is malformed, fall back to defaults.
            pass
    return [(re.compile(pat, fl), repl) for pat, repl, fl in _DEFAULT_TOKEN_FIXES]


_TOKEN_FIXES: list[tuple[re.Pattern[str], str]] = _load_token_fixes()

# Light lexicon for spotting “mirrored” PDF text (some figures/tables store glyphs reversed).
_COMMON_WORDS: frozenset[str] = frozenset(
    """
    the a an and or of to in for on at by with from is are was were be been being
    have has had do does did will would could should may might must shall not but if
    when where than then so as at off out up down over all any both each few more most
    other some such no nor too very can just than into through during before after
    above below between under again further once here there when why how all both
    this that these those it its we our you your they their what which who whom
    study studies patient patients result results method methods material materials
    abstract introduction discussion conclusion conclusions reference references
    figure figures table tables supplementary appendix data analysis analyses model
    models human humans review reviews screening precision recall sensitivity
    specificity accuracy balanced performance ensemble configuration configurations
    prompt prompts parallel cross dataset development comprehensive replication
    replicated search searches workflow optimal median mean total group groups arm
    arms treatment control trial trials clinical random randomized outcome outcomes
    primary secondary endpoint endpoints intervention comparator included exclusion
    criteria objective background methods results findings conclusion funding
    author authors objective objectives design sample size power statistical
    significant significance confidence interval hazard ratio odds ratio
    sensitivity specificity positive negative predictive value auc roc
    gpt llm llms turbo large language model models machine learning automated
    systematic cochrane library performance variable variables describing schema
    parallel heavy extreme stone gpt maintaining calculated original compared
    within between across per year years month day days week weeks
    every reviewer reviewers volume vol issue page pages journal
    march april january february may june july august september october november december
    doi http https org com edu
    """.split()
)


def _line_english_word_score(line: str) -> float:
    words = re.findall(r"[A-Za-z]+", line.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _COMMON_WORDS)
    return hits / len(words)


def _should_unflip_line(line: str) -> bool:
    s = line.strip()
    # Single-token lines (after reflow) e.g. "yreve" → "every"
    if 5 <= len(s) <= 13 and re.fullmatch(r"[A-Za-z]+", s):
        rev = s[::-1]
        if rev.lower() in _COMMON_WORDS and s.lower() not in _COMMON_WORDS:
            return True
    if len(s) < 14:
        return False
    if not re.search(r"[A-Za-z]{4,}", s):
        return False
    rev = s[::-1]
    sl = _line_english_word_score(s)
    sr = _line_english_word_score(rev)
    # Reversed line reads like English; forward line does not (typical figure/table garbage).
    if sr >= 0.14 and sr > sl + 0.12:
        return True
    # Strong signal: “Figure” / “Table” backwards etc.
    if re.search(r"(erugiF|elbaT|stluseR|dohteM|tcurtsnoc)", s) and sr > sl:
        return True
    return False


def fix_reversed_pdf_text(text: str) -> str:
    """
    Fix lines that are stored in reverse order in the PDF (common in some figures).
    """
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        if _should_unflip_line(line):
            lead = len(line) - len(line.lstrip())
            trail = len(line) - len(line.rstrip())
            flipped = line.strip()[::-1]
            out.append(" " * lead + flipped + " " * trail)
        else:
            out.append(line)
    return "\n".join(out)


def collapse_spaced_letter_runs(text: str) -> str:
    """
    Collapse “J o u r n a l” → “Journal” when the PDF split one word into spaced glyphs.
    Only merges runs of 1-char tokens (last token may be 2 chars like “al”).
    """
    def fix_line(line: str) -> str:
        parts = line.split()
        if len(parts) < 6:
            return line
        new_parts: list[str] = []
        i = 0
        while i < len(parts):
            if len(parts[i]) != 1 or not parts[i].isalpha():
                new_parts.append(parts[i])
                i += 1
                continue
            j = i
            buf: list[str] = []
            while j < len(parts):
                q = parts[j]
                if len(q) == 1 and q.isalpha():
                    buf.append(q)
                    j += 1
                    continue
                if len(q) == 2 and q.isalpha() and len(buf) >= 3:
                    buf.append(q)
                    j += 1
                    break
                break
            merged = "".join(buf)
            if len(buf) >= 4 and len(merged) >= 5:
                new_parts.append(merged)
                i = j
            else:
                new_parts.append(parts[i])
                i += 1
        return " ".join(new_parts)

    return "\n".join(fix_line(ln) for ln in text.split("\n"))


def estimate_reversed_line_ratio(text: str) -> float:
    """Share of non-empty lines that look like reversed Latin (for extractor fallback)."""
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    if not lines:
        return 0.0
    bad = sum(1 for ln in lines if _should_unflip_line(ln))
    return bad / len(lines)


def remove_invisible_chars(text: str) -> str:
    """Soft hyphens, ZWSP, BOM, word joiners — common in publisher PDFs."""
    for ch in "\u200b\u200c\u200d\u2060\ufeff\u00ad":
        text = text.replace(ch, "")
    return text


def normalize_hyphen_chars(text: str) -> str:
    """Map Unicode hyphens/dashes to ASCII so reflow patterns match."""
    for ch in "\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d":
        text = text.replace(ch, "-")
    return text


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def merge_hyphenated_linebreaks(text: str) -> str:
    """
    Join words split across lines with a hyphen (myo-cardial → myocardial).

    Short prefixes are often syllable breaks → merge without hyphen, except when
    that would break real compounds (as-treated, to-treat) or acronyms (STS-PROM).
    """
    changed = True
    while changed:
        changed = False

        # ALL-CAPS / acronym line breaks: STS-\nPROM → STS-PROM (before len≤3 merge)
        n0 = re.sub(
            r"([A-Z]{2,})-\s*\n\s*([A-Z][A-Za-z0-9]*)",
            r"\1-\2",
            text,
        )
        if n0 != text:
            text = n0
            changed = True
            continue

        def _join_lower(m: re.Match[str]) -> str:
            prefix, suffix = m.group(1), m.group(2)
            pl, sl = prefix.lower(), suffix.lower()
            # Keep hyphen for common compounds that are not syllable breaks
            if pl == "as" and sl == "treated":
                return f"{prefix}-{suffix}"
            if pl == "to" and sl == "treat":
                return f"{prefix}-{suffix}"
            if pl == "as" and sl == "signed":
                return prefix + suffix
            if len(prefix) <= 3:
                return prefix + suffix
            return f"{prefix}-{suffix}"

        n = re.sub(r"([A-Za-z0-9]+)-\s*\n\s*([a-z][a-z0-9]*)", _join_lower, text)
        if n != text:
            text = n
            changed = True
            continue

        n2 = re.sub(r"([A-Za-z0-9]+)-\s*\n\s*([A-Z0-9][A-Za-z0-9]*)", r"\1-\2", text)
        if n2 != text:
            text = n2
            changed = True

    for pat, repl in (
        (re.compile(r"inter-\s*mediate", re.IGNORECASE), "intermediate"),
        (re.compile(r"semi-\s*random", re.IGNORECASE), "semirandom"),
    ):
        text = pat.sub(repl, text)

    # Recovery when reflow still leaves common glued tokens
    for pat, repl in (
        (re.compile(r"\bKaplanMeier\b"), "Kaplan–Meier"),
        (re.compile(r"\bintentiontotreat\b", re.IGNORECASE), "intention-to-treat"),
        (re.compile(r"\bastreated\b", re.IGNORECASE), "as-treated"),
        (re.compile(r"\bcausespecific\b", re.IGNORECASE), "cause-specific"),
        (re.compile(r"\bas-treat-ed\b", re.IGNORECASE), "as-treated"),
    ):
        text = pat.sub(repl, text)
    return text


def fix_single_letter_linebreaks(text: str) -> str:
    """Repair 'T ranscatheter' when a single capital letter was split from the rest."""
    # Do not merge 'I word' / 'A word' (pronouns / articles)
    def _merge(m: re.Match[str]) -> str:
        letter, rest = m.group(1), m.group(2)
        if letter in "AI":
            return m.group(0)
        return letter + rest

    return re.sub(
        r"(?<![A-Za-z])([B-HJ-Z])\s+([a-z]{5,})",
        _merge,
        text,
    )


def _is_standalone_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _PAGE_HEADER_LINE.match(s):
        return True
    if s in _KNOWN_HEADINGS:
        return True
    # Short ALL-CAPS line (typical section title); skip single-letter lines (PDF glitches)
    if s.isupper() and len(s) > 1 and len(s) < 90 and len(s.split()) <= 8:
        return True
    return False


def _should_merge_wrapped_line(prev: str, nxt: str) -> bool:
    """Merge a line break when the next line continues the same paragraph."""
    if not prev.strip() or not nxt.strip():
        return False
    if _is_standalone_heading(prev):
        return False
    first = nxt.lstrip()[0]
    if not first.islower():
        return False
    # Sentence end on previous line → usually a new sentence (starts with capital on next line)
    p = prev.rstrip()
    if not p:
        return False
    if p[-1] in ".?!":
        # Allow merge after "e.g." / "i.e." / "vs." / "Dr."
        lower = p.lower()
        for suf in ("e.g.", "i.e.", "etc.", "vs.", "al.", "no.", "fig.", "dr.", "mr.", "mrs."):
            if lower.endswith(suf):
                return True
        return False
    return True


def merge_wrapped_lines(text: str) -> str:
    """Join single newlines inside paragraphs; keep breaks before headings / after blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        while i + 1 < len(lines) and _should_merge_wrapped_line(cur, lines[i + 1]):
            cur = cur.rstrip() + " " + lines[i + 1].strip()
            i += 1
        out.append(cur)
        i += 1
    return "\n".join(out)


def apply_token_fixes(text: str) -> str:
    for pat, repl in _TOKEN_FIXES:
        text = pat.sub(repl, text)
    return text


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Preserve paragraph breaks; collapse spaces within lines
    parts = text.split("\n")
    parts = [re.sub(r"[ \t]+", " ", p) for p in parts]
    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_footer_noise_line(s: str) -> bool:
    """True if this line is trailing publisher junk (used only when stripping from page end)."""
    t = s.strip()
    if not t:
        return True
    if _PAGE_NUM_LINE.match(t) or _DIGITS_ONLY.match(t) or _PAGE_NUM_PADDED.match(t):
        return True
    if len(t) <= 2 and t.isdigit():
        return True
    for pat in _BOILERPLATE_LINE_PATTERNS:
        if pat.search(t):
            return True
    return False


def _strip_trailing_boilerplate_lines(lines: list[str]) -> list[str]:
    out = list(lines)
    while out and _is_footer_noise_line(out[-1]):
        out.pop()
    return out


def _normalize_line_for_repetition(s: str) -> str:
    """
    Normalize a candidate footer line for repeat detection.
    """
    t = remove_invisible_chars(s)
    t = normalize_unicode(t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 140:
        t = t[:140]
    return t.lower()


def strip_repeating_tail_lines_from_pages(
    raw_pages: list[str],
    tail_line_budget: int = 10,
    min_pages: int = 2,
) -> list[str]:
    """
    Remove footer-like lines that repeat across multiple pages.

    This is a publisher-agnostic approach that helps strip things like
    download/disclosure/copyright blocks when we don't have exact patterns.
    """
    normalized_tail_lines: list[list[str]] = []
    counter: dict[str, int] = {}

    for raw in raw_pages:
        t = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln for ln in t.split("\n") if ln.strip()]
        tail = lines[-tail_line_budget:] if len(lines) > tail_line_budget else lines
        norms: list[str] = []
        for ln in tail:
            ln_stripped = ln.strip()
            if _PAGE_NUM_LINE.match(ln_stripped) or _DIGITS_ONLY.match(ln_stripped) or _PAGE_NUM_PADDED.match(ln_stripped):
                continue
            norm = _normalize_line_for_repetition(ln_stripped)
            if not norm:
                continue
            norms.append(norm)
            counter[norm] = counter.get(norm, 0) + 1
        normalized_tail_lines.append(norms)

    repeated = {line for line, c in counter.items() if c >= min_pages and len(line) >= 6}
    if not repeated:
        return raw_pages

    cleaned_pages: list[str] = []
    for raw in raw_pages:
        t = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = t.split("\n")

        # Trim trailing empty lines first
        while lines and not lines[-1].strip():
            lines.pop()

        # Pop repeated/footer noise from the bottom
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if _is_footer_noise_line(last):
                lines.pop()
                continue
            if _normalize_line_for_repetition(last) in repeated:
                lines.pop()
                continue
            break

        cleaned_pages.append("\n".join(lines))

    return cleaned_pages


def strip_repeating_head_lines_from_pages(
    raw_pages: list[str],
    head_line_budget: int = 8,
    min_pages: int = 2,
) -> list[str]:
    """
    Remove header-like lines that repeat across multiple pages.
    This complements `strip_repeating_tail_lines_from_pages()` which removes
    footer-like lines.
    """
    # Build repetition counts from per-page top slices
    counter: dict[str, int] = {}
    page_norm_heads: list[list[str]] = []

    for raw in raw_pages:
        t = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln for ln in t.split("\n") if ln.strip()]
        head = lines[:head_line_budget] if len(lines) > head_line_budget else lines
        norms: list[str] = []
        for ln in head:
            ln_stripped = ln.strip()
            if _PAGE_NUM_LINE.match(ln_stripped) or _DIGITS_ONLY.match(ln_stripped) or _PAGE_NUM_PADDED.match(ln_stripped):
                continue
            if _DIGITS_PIPE.match(ln_stripped):
                continue
            norm = _normalize_line_for_repetition(ln_stripped)
            if not norm:
                continue
            norms.append(norm)
            counter[norm] = counter.get(norm, 0) + 1
        page_norm_heads.append(norms)

    repeated = {line for line, c in counter.items() if c >= min_pages and len(line) >= 6}
    if not repeated:
        return raw_pages

    cleaned_pages: list[str] = []
    for raw in raw_pages:
        t = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = t.split("\n")
        # Trim leading empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        # Remove repeated lines from the top until we hit non-repeated content
        while lines and lines[0].strip():
            first = lines[0].strip()
            if _PAGE_NUM_LINE.match(first) or _DIGITS_ONLY.match(first) or _PAGE_NUM_PADDED.match(first):
                lines.pop(0)
                continue
            if _DIGITS_PIPE.match(first):
                lines.pop(0)
                continue
            if _normalize_line_for_repetition(first) in repeated:
                lines.pop(0)
                continue
            break
        cleaned_pages.append("\n".join(lines))
    return cleaned_pages


def clean_page_raw_text(raw: str, page_num: int, total_pages: int) -> str:
    """
    Per-page pass before reflow: drop publisher footers / watermarks and optional
    NEJM promo headers. Page-number artifacts are removed later in the global cleaner.
    """
    t = _LEADING_PROMO_PREFIX.sub("", raw, count=1)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    lines = _strip_trailing_boilerplate_lines(lines)
    body = "\n".join(lines).strip()
    return body


def strip_likely_noise_lines(text: str) -> str:
    """Drop isolated page-number lines and very short digit-only lines."""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append("")
            continue
        if _PAGE_NUM_LINE.match(s) or _DIGITS_ONLY.match(s) or _PAGE_NUM_PADDED.match(s):
            continue
        if _DIGITS_PIPE.match(s):
            continue
        if len(s) <= 2 and s.isdigit():
            continue
        out.append(line.rstrip())
    return "\n".join(out)


def strip_known_boilerplate_lines(text: str) -> str:
    """
    Remove known publisher boilerplate lines anywhere in the document.

    We apply this after reflow so strings like "Disclosure forms..." are contiguous.
    """
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        if any(p.search(line) for p in _BOILERPLATE_LINE_PATTERNS):
            continue
        out.append(line.rstrip())
    return "\n".join(out)


def clean_full_text(raw: str) -> str:
    """
    Full pipeline: invisible chars → unicode → hyphen chars → hyphen+linebreak
    reflow → wrapped-line join → single-letter fix → token fixes → noise → whitespace.


    """
    t = remove_invisible_chars(raw)
    t = normalize_unicode(t)
    t = normalize_hyphen_chars(t)
    t = merge_hyphenated_linebreaks(t)
    t = merge_wrapped_lines(t)
    t = fix_single_letter_linebreaks(t)
    # After reflow, mirrored figure/table strings often sit on one line and can flip.
    t = fix_reversed_pdf_text(t)
    t = collapse_spaced_letter_runs(t)
    t = apply_token_fixes(t)
    t = strip_known_boilerplate_lines(t)
    t = strip_likely_noise_lines(t)
    t = normalize_whitespace(t)
    return t
