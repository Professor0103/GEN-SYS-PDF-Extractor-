"""
Parse author names and build short citation strings (Last; Last and Last; Last et al.).
"""

from __future__ import annotations

import re
from typing import Any

_DEGREE_TOK = re.compile(
    r"^(MA|MD|PhD|MBBS|BSc|BCh|BChir|BA|MS|MBA|MPH|RN|DO|DPhil|FRCP|FRCS)\d?[\d,]*$",
    re.I,
)
_NUM_TOK = re.compile(r"^\d")


def _split_person_name(segment: str) -> tuple[str, str] | None:
    s = re.sub(r"\s+", " ", segment.strip())
    if not s or len(s) < 2:
        return None
    if "@" in s or "http" in s.lower():
        return None
    if _DEGREE_TOK.match(s) or _NUM_TOK.match(s):
        return None
    parts = s.split()
    if len(parts) == 1:
        return (parts[0], parts[0])
    return (" ".join(parts[:-1]), parts[-1])


def _parse_comma_author_line(line: str) -> list[tuple[str, str]]:
    raw = [p.strip() for p in line.split(",") if p.strip()]
    out: list[tuple[str, str]] = []
    for seg in raw:
        if _DEGREE_TOK.match(seg) or _NUM_TOK.match(seg):
            continue
        if len(seg) < 2:
            continue
        pair = _split_person_name(seg)
        if pair:
            out.append(pair)
    return out


def parse_authors_from_pdf_metadata(author_field: str) -> list[tuple[str, str]]:
    if not author_field or not str(author_field).strip():
        return []
    text = str(author_field).replace("\n", ";")
    out: list[tuple[str, str]] = []
    for chunk in re.split(r"[;]", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "," in chunk and not re.search(r"\d", chunk.split(",")[0]):
            # "Lastname, Firstname" or "Lastname, First M."
            left, right = chunk.split(",", 1)
            left, right = left.strip(), right.strip()
            if left and right:
                fn = right.split()[0] if right else left
                ln = left.split()[-1] if left else right
                out.append((fn, ln))
                continue
        pair = _split_person_name(chunk)
        if pair:
            out.append(pair)
    return out[:40]


def _find_author_block_lines(head: str) -> str:
    """Text before Abstract / Objective / DOI / BACKGROUND where authors often sit."""
    lines = head.replace("\r\n", "\n").split("\n")
    buf: list[str] = []
    for line in lines[:60]:
        s = line.strip()
        low = s.lower()
        if re.match(r"^(abstract|introduction|background|objective|methods?)\s*:?\s*$", low):
            break
        if low.startswith("doi:") or "doi.org" in low:
            break
        if low.startswith("http"):
            break
        buf.append(s)
    return " ".join(buf)


def parse_authors_from_text_head(text: str) -> list[tuple[str, str]]:
    head = text[:12000]
    block = _find_author_block_lines(head)
    if not block:
        return []
    # Prefer a substring with several commas (author list)
    if block.count(",") >= 2:
        pairs = _parse_comma_author_line(block)
        if len(pairs) >= 1:
            return pairs[:40]
    # Single-line style: "Name Name, Name Name, ..."
    for line in head.split("\n")[:40]:
        s = line.strip()
        if s.count(",") >= 2 and len(s) < 1200:
            pairs = _parse_comma_author_line(s)
            if len(pairs) >= 1:
                return pairs[:40]
    return []


def merge_authors_into_metadata(
    meta: dict[str, Any],
    pruned_text: str,
    pdf_metadata: dict[str, Any] | None,
) -> None:
    pdf_meta = pdf_metadata or {}
    author_raw = (
        pdf_meta.get("author")
        or pdf_meta.get("Author")
        or pdf_meta.get("authors")
        or ""
    )
    authors = parse_authors_from_pdf_metadata(str(author_raw))
    if len(authors) < 1:
        authors = parse_authors_from_text_head(pruned_text)
    meta["authors"] = [{"first": f, "last": l} for f, l in authors]


def _norm_key_first(first: str) -> str:
    return first.lower().strip().rstrip(".")


def _display_last(last: str) -> str:
    return last.strip()


def format_citation_label(
    authors: list[dict[str, str]],
    year: int | None,
    year_letter_suffix: str = "",
) -> str:
    """
    One author: Last [year]
    Two authors: Last1 and Last2 [year]
    Three or more: Last1 et al. [year]
    year_letter_suffix: 'a', 'b', ... appended inside brackets to the year when batch-disambiguating.
    """
    ypart = "n.d."
    if year is not None:
        ypart = f"{year}{year_letter_suffix}" if year_letter_suffix else str(year)

    if not authors:
        return f"[{ypart}]"

    lasts = [_display_last(a["last"]) for a in authors if a.get("last")]
    lasts = [x for x in lasts if x]
    if not lasts:
        return f"[{ypart}]"

    if len(lasts) == 1:
        body = f"{lasts[0]}"
    elif len(lasts) == 2:
        body = f"{lasts[0]} and {lasts[1]}"
    else:
        body = f"{lasts[0]} et al."

    return f"{body} [{ypart}]"


def citation_disambiguation_key(
    authors: list[dict[str, str]],
    year: int | None,
) -> tuple[str, str, int] | None:
    """First author (normalized) + year for batch duplicate detection."""
    if not authors or year is None:
        return None
    a0 = authors[0]
    f, l = a0.get("first", ""), a0.get("last", "")
    if not l:
        return None
    return (_norm_key_first(f or ""), l.lower().strip(), int(year))


def assign_citation_suffixes(records: list[dict[str, Any]]) -> None:
    """
    When the same first author + last name + publication year appears more than once
    in a batch, use [yeara], [yearb], ...
    """
    from collections import defaultdict

    groups: defaultdict[tuple[str, str, int], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        meta = rec.get("metadata") or {}
        authors = meta.get("authors") or []
        year = meta.get("year")
        key = citation_disambiguation_key(authors, year)
        if key is None:
            rec["citation_year_suffix"] = ""
            continue
        groups[key].append(i)

    for key, indices in groups.items():
        if len(indices) <= 1:
            for idx in indices:
                records[idx]["citation_year_suffix"] = ""
            continue
        for j, idx in enumerate(indices):
            # first duplicate pair -> a, b; three -> a, b, c
            records[idx]["citation_year_suffix"] = chr(ord("a") + j)


def build_metadata_citation_label(meta: dict[str, Any], year_suffix: str = "") -> str:
    authors = meta.get("authors") or []
    year = meta.get("year")
    return format_citation_label(authors, year, year_suffix)
