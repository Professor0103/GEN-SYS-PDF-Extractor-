"""
Heuristic section segmentation for academic papers (full text).

Not perfect across all publishers; prioritizes recall for LLM downstream use.
"""

from __future__ import annotations

import re
from typing import Any


_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"(?im)^\s*(abstract)\s*$", re.MULTILINE)),
    ("introduction", re.compile(r"(?im)^\s*(?:1\.?\s*)?(introduction)\s*$")),
    ("methods", re.compile(r"(?im)^\s*(?:2\.?\s*)?(methods?|materials?\s+and\s+methods?|patients?\s+and\s+methods?)\s*$")),
    ("results", re.compile(r"(?im)^\s*(?:3\.?\s*)?(results?|findings?)\s*$")),
    ("discussion", re.compile(r"(?im)^\s*(?:4\.?\s*)?(discussion)\s*$")),
    ("references", re.compile(r"(?im)^\s*(references|bibliography)\s*$", re.MULTILINE)),
]


def _split_by_headings(text: str) -> dict[str, str]:
    """Find first occurrence of each standard heading; assign text between them."""
    sections: dict[str, str] = {
        "abstract": "",
        "introduction": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "references": "",
    }
    hits: list[tuple[int, str, re.Match[str]]] = []
    for name, pat in _SECTION_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append((m.start(), name, m))
    hits.sort(key=lambda x: x[0])

    if not hits:
        return sections

    for i, (_, name, m) in enumerate(hits):
        start = m.end()
        end = hits[i + 1][2].start() if i + 1 < len(hits) else len(text)
        chunk = text[start:end].strip()
        if chunk and (not sections[name] or len(chunk) > len(sections[name])):
            sections[name] = chunk
    return sections


def guess_title_first_lines(text: str, max_lines: int = 8) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    # Skip common journal/header lines
    skip_prefixes = ("doi:", "http", "www.", "copyright", "©", "received ", "accepted ")
    title_lines: list[str] = []
    for ln in lines[:max_lines]:
        low = ln.lower()
        if any(low.startswith(p) for p in skip_prefixes):
            continue
        if len(ln) > 200:
            break
        title_lines.append(ln)
        if len(title_lines) >= 3:
            break
    raw = " ".join(title_lines[:2]) if title_lines else lines[0][:500]
    return raw[:220].rstrip()


def extract_metadata_light(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"title": "", "authors": [], "year": None}
    meta["title"] = guess_title_first_lines(text)
    year_m = re.search(r"\b(19|20)\d{2}\b", text[:2500])
    if year_m:
        try:
            meta["year"] = int(year_m.group(0))
        except ValueError:
            pass
    return meta


def segment_sections(clean_text: str) -> dict[str, str]:
    return _split_by_headings(clean_text)


def chunk_text(
    text: str,
    max_chars: int = 48_000,
    overlap: int = 800,
) -> list[str]:
    """
    Prefer paragraph boundaries (like the paper); only split hard when a block
    exceeds max_chars. Overlap applies between adjacent chunks when splitting.
    """
    if not text or max_chars <= 0:
        return []
    # Paragraphs: blank-line separated (matches reflowed clean_text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush_buf() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for p in paras:
        plen = len(p) + (2 if buf else 0)  # "\n\n" join
        if plen > max_chars:
            flush_buf()
            # Oversized paragraph: fall back to sliding window
            i = 0
            n = len(p)
            while i < n:
                end = min(i + max_chars, n)
                chunks.append(p[i:end])
                if end >= n:
                    break
                i = max(0, end - overlap)
            continue
        if buf_len + plen <= max_chars:
            buf.append(p)
            buf_len += plen
        else:
            flush_buf()
            buf.append(p)
            buf_len = len(p)

    flush_buf()
    return chunks
