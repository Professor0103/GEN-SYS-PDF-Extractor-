"""
Sanitize text for the LLM-facing export: drop reference lists, repeated journal
headers, and other PDF chrome that does not help screening.
"""

from __future__ import annotations

import re
from typing import Any


def _looks_like_journal_or_issue_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    low = t.lower()
    if "doi.org" in low or low.startswith("http://") or low.startswith("https://"):
        return True
    if "advance access publication" in low:
        return True
    if re.search(r"\bjournal of\b", low) and (
        "vol." in low or "issue" in low or re.search(r"\b20\d{2}\b", t) or "(" in t and ")" in t
    ):
        return True
    if re.match(r"^n engl j med\b", low):
        return True
    if re.match(r"^received:\s*", low) and "editorial decision" in low:
        return True
    if re.match(r"^accepted:\s*", low):
        return True
    if re.match(r"^copyright\b", low) or "creative commons" in low:
        return True
    if "©" in t or "\u00a9" in t:
        return True
    if "published by oxford" in low or (
        "published by" in low and "university press" in low
    ):
        return True
    if "the author(s)" in low and "published" in low:
        return True
    if re.match(r"^page\s+\d+", low):
        return True
    # Footer: volume + page + journal name on one line
    if re.match(r"^\d{3,4}\s+Journal of\b", t, re.I):
        return True
    if re.match(r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s*$", low):
        return True
    if re.match(r"^,?\s*march\s*$", low):
        return True
    # Journal site promo + DOI (e.g. AHA)
    if "is available at" in low and re.search(r"https?://", t):
        return True
    if "circulation research" in low and "ahajournals.org" in low:
        return True
    # Page + journal + issue date (e.g. 722 Circulation Research August 31, 2018)
    if re.match(
        r"^\d{3,4}\s+.+\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s*$",
        t,
        re.I,
    ):
        return True
    if re.match(r"^\d{3,4}\s+Circulation Research\b", t, re.I):
        return True
    # Running header with page (e.g. "Wang et al Cardiovascular GPCRs 717")
    if (
        len(t) <= 110
        and re.search(r"\bet al\s+[A-Z][a-zA-Z\-]+\s+", t)
        and re.search(r"\d{3,4}\s*$", t)
    ):
        return True
    # Column bleed: lone page index
    if re.match(r"^\s*\d{1,3}\s*,\s*$", t):
        return True
    # Author-address blocks that land in the body (not only leading peel)
    if re.match(r"^from the department of\b", low):
        return True
    if re.match(r"^correspondence to\b", low):
        return True
    # Zip + email fragment left after stripping "Correspondence to …"
    if re.match(r"^\s*\d{5}\.?\s+Email\s+\S+@\S+", t, re.I):
        return True
    # Wrapped affiliation line: "(H.A.R.), Duke University Medical Center, Durham, NC."
    if (
        len(t) < 240
        and re.match(r"^\([^)]+\),\s*.+\b(?:University|Medical Center)\b", t)
    ):
        return True
    return False


def _looks_like_section_heading_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if t.isupper() and len(t) < 90 and len(t.split()) <= 10:
        return True
    if re.match(
        r"^(abstract|introduction|methods?|results?|discussion|conclusions?|references|bibliography)\s*:?\s*$",
        t,
        re.I,
    ):
        return True
    return False


def _looks_like_article_title_candidate(s: str) -> bool:
    t = s.strip()
    if len(t) < 14 or len(t) > 280:
        return False
    low = t.lower()
    if "doi.org" in low or "http" in low:
        return False
    if re.match(r"^journal of\b", low):
        return False
    if re.match(r"^(research and applications|original article|review article)\s*$", low, re.I):
        return False
    words = t.split()
    if len(words) < 3:
        return False
    if all(w.isupper() for w in words if w.isalpha()) and len(t) < 100:
        return False
    return True


def _current_title_is_bibliographic_chrome(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    low = t.lower()
    if "doi.org" in low or "http" in low:
        return True
    if "advance access" in low:
        return True
    if "journal of" in low and re.search(r"\b20\d{2}\b", t):
        return True
    if re.search(r"\bvol\.?\s*\d+", low) and "journal" in low:
        return True
    return False


def refine_llm_title(clean_text: str, metadata_title: str) -> str:
    """
    Prefer the real article title over journal name / DOI / issue lines that
    guess_title_first_lines often picks up.
    """
    cur = (metadata_title or "").strip()
    if not _current_title_is_bibliographic_chrome(cur):
        return cur[:280]

    head = clean_text[:8000] if clean_text else ""
    for line in head.split("\n"):
        s = line.strip()
        if not s:
            continue
        if _looks_like_journal_or_issue_line(s):
            continue
        if _looks_like_section_heading_line(s):
            continue
        if _looks_like_article_title_candidate(s):
            return s[:280]
    return cur[:280]


def _is_prose_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if len(t) > 100 and t.endswith("."):
        return True
    if re.match(
        r"^(Objective|Background|Methods|Results|Conclusion|Abstract|Introduction)\s*:",
        t,
        re.I,
    ):
        return True
    if re.match(r"^(The|This|We|In|Here|As a|However|Although)\s+", t) and len(t) > 50:
        return True
    return False


def _looks_like_author_or_affiliation_line(s: str) -> bool:
    t = s.strip()
    if not t or len(t) > 400:
        return False
    if re.search(
        r",\s*(MA|MD|PhD|MBBS|BSc|BCh|BChir|BA|MS|M\.A\.|B\.A\.)\b",
        t,
        re.I,
    ):
        return True
    if re.search(
        r"\b(University|Hospital|Foundation Trust|NHS|Department of|Medical Center|"
        r"Clinical Neurosciences|Graduate School|Cambridge|Oxford)\b",
        t,
        re.I,
    ):
        return True
    if re.match(r"^\d+[A-Za-z]", t) and ("," in t or "@" in t):
        return True
    if re.match(r"^\*?\s*A complete list of", t, re.I):
        return True
    if "@" in t and ".edu" in t.lower():
        return True
    return False


def strip_leading_author_affiliation_lines(text: str, max_peel: int = 24) -> str:
    """Remove author/address lines often repeated at the top of abstract/intro."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    peeled = 0
    while i < len(lines) and peeled < max_peel:
        s = lines[i]
        st = s.strip()
        if not st:
            i += 1
            continue
        if _is_prose_line(st):
            break
        if _looks_like_author_or_affiliation_line(st):
            i += 1
            peeled += 1
            continue
        if len(st) < 180 and st.count(",") >= 4 and re.search(
            r"(Dr\.|Prof\.|MD|PhD|MBBS)", st, re.I
        ):
            i += 1
            peeled += 1
            continue
        break
    return "\n".join(lines[i:]).lstrip()


def filter_llm_section_body(text: str) -> str:
    """Drop repeated journal/footer lines and similar noise inside a section."""
    if not text.strip():
        return text
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    for line in lines:
        if _looks_like_journal_or_issue_line(line):
            continue
        out.append(line.rstrip())
    text2 = "\n".join(out)
    text2 = re.sub(r"\n{3,}", "\n\n", text2)
    return text2.strip()


def references_placeholder() -> str:
    return (
        "[References omitted for screening — full bibliography is in the source PDF / "
        "full-text export if you need it.]"
    )


def sanitize_sections_for_llm(sections: dict[str, str]) -> dict[str, str]:
    """Return a copy of sections with bodies filtered; references replaced."""
    out: dict[str, str] = {}
    for k, v in sections.items():
        if k == "references":
            out[k] = ""
            continue
        if isinstance(v, str) and v.strip():
            body = v
            if k in ("abstract", "introduction"):
                body = strip_leading_author_affiliation_lines(body)
            body = filter_llm_section_body(body)
            out[k] = strip_reference_tail_for_llm(body)
        else:
            out[k] = v or ""
    return out


# "References" merged with the next token on the same line (column layout)
_REF_HEADING_FRAGMENT = re.compile(
    r"(?im)^\s*references\s+(?!\s*to\s)(?=\S)",
)
# Standalone section headings near the end of the manuscript
_LATE_END_MATTER = re.compile(
    r"(?im)^\s*(?:"
    r"funding|"
    r"author contributions|"
    r"competing interests|"
    r"conflict[s]? of interest|"
    r"acknowledg(?:e)?ments|"
    r"references cited|"
    r"supplementary material\b"
    r")\s*$",
)
# Numbered bibliography lines that bled into the body before a heading matched
_REF_NUMBERED_ENTRY = re.compile(
    r"(?m)^\s*\d{1,3}\.\s+[A-Z][a-zA-Z'\-]+ [A-Z][A-Za-z'\-]*,.*\b(?:"
    r"Cochrane Database|doi\.org|https?://(?:www\.)?doi|JAMA Netw|"
    r"Lancet|BMJ|PubMed|Embase|Ann Intern Med|PLoS|PLOS ONE|"
    r"Nat Mach|arXiv|J Med Internet|J Clin Epidemiol|J Nurs\.|"
    r"Stud Health Technol|Eur Heart J|N Engl J Med|Circulation|"
    r"J Am Coll Cardiol|Ann Thorac Surg|Nat Med|Nat Commun"
    r")\b",
)


def _strip_trailing_publisher_footer_lines(text: str) -> str:
    """Drop trailing copyright or publisher taglines."""
    lines = text.split("\n")
    footer_res = (
        re.compile(r"^©|^\s*©", re.I),
        re.compile(r"the author\(s\).{0,120}published", re.I),
        re.compile(r"published by .{0,80}university press", re.I),
        re.compile(r"^\s*research and applications\s*$", re.I),
    )
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if any(p.search(last) for p in footer_res):
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


def strip_reference_tail_for_llm(text: str) -> str:
    """Remove reference lists and common late PDF junk from the tail of the text."""
    if not text.strip():
        return text
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    cut_at: int | None = None

    for pat in (
        r"(?im)^\s*references\s*$",
        r"(?im)^\s*bibliography\s*$",
    ):
        m = re.search(pat, t)
        if m is not None:
            cut_at = m.start() if cut_at is None else min(cut_at, m.start())

    m = _REF_HEADING_FRAGMENT.search(t)
    if m is not None:
        cut_at = m.start() if cut_at is None else min(cut_at, m.start())

    m = _LATE_END_MATTER.search(t)
    if m is not None and m.start() >= min(8000, max(2000, len(t) // 5)):
        cut_at = m.start() if cut_at is None else min(cut_at, m.start())

    m = _REF_NUMBERED_ENTRY.search(t)
    if m is not None:
        # Avoid rare early-document false positives; reference lists are almost always
        # in the latter part of the extracted text.
        if m.start() >= min(4000, max(0, len(t) // 4)):
            cut_at = m.start() if cut_at is None else min(cut_at, m.start())

    if cut_at is not None:
        t = t[:cut_at].rstrip()

    t = _strip_trailing_publisher_footer_lines(t)
    return t


def normalize_llm_text_spacing(text: str) -> str:
    """
    Tidy whitespace for the LLM export: strip trailing spaces per line, cap blank
    runs between paragraphs, and remove any stray [LLM SIGNPOST: ...] lines.
    """
    if not text:
        return text
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(?m)^\s*\[LLM SIGNPOST:[^\n]*\n?", "", t)
    lines = [ln.rstrip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def build_llm_ready_payload(
    record: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    """Title and section bodies for the LLM text export."""
    clean_text = (record.get("clean_text") or "").strip()
    meta = record.get("metadata") or {}
    sections = dict(record.get("sections") or {})

    title = refine_llm_title(clean_text, meta.get("title") or "")
    sections_llm = sanitize_sections_for_llm(sections)
    return title, sections_llm
