"""Orchestrate classify → extract → clean → segment → schema output."""

from __future__ import annotations

import uuid
import re
from pathlib import Path
from typing import Any

from sr_pipeline.authors import (
    assign_citation_suffixes,
    build_metadata_citation_label,
    merge_authors_into_metadata,
)
from sr_pipeline.classifier import is_likely_scanned
from sr_pipeline.cleaner import clean_full_text
from sr_pipeline.extractor import extract_full_text
from sr_pipeline.exporter import write_txt, zip_outputs
from sr_pipeline.llm_normalize import (
    build_llm_ready_payload,
    filter_llm_section_body,
    normalize_llm_text_spacing,
    refine_llm_title,
    strip_reference_tail_for_llm,
)
from sr_pipeline.segmenter import chunk_text, extract_metadata_light, segment_sections


def _year_from_pdf_metadata(pdf_metadata: dict[str, Any] | None) -> int | None:
    """Best-effort year from PyMuPDF's document metadata (creationDate / modDate)."""
    if not pdf_metadata:
        return None
    for key in ("creationDate", "modDate", "CreationDate", "ModDate"):
        raw = str(pdf_metadata.get(key) or "")
        m = re.search(r"(19|20)(\d{2})", raw)
        if m:
            try:
                return int(m.group(1) + m.group(2))
            except ValueError:
                continue
    return None


def _extract_captions_and_prune(text: str) -> tuple[list[str], list[str], str]:
    """
    Extract Figure/Table captions and remove them from the main text used for
    section segmentation (so they don't appear twice).

    A caption line looks like ``Figure 1. ...`` / ``Table 2: ...``. We greedily
    consume continuation lines, but with hard caps so we never swallow body
    prose: continuation stops at the first sentence-ending line, at the caption
    line budget, or as soon as we see a paragraph break.
    """
    lines = text.split("\n")

    # A real caption start is "Figure 1.", "Fig 2", "Table 3:", etc. The negative
    # lookahead rejects inline references like "(Fig. 4B), with..." which are body
    # prose where a stray "(" landed on the previous line. We also require some
    # text content to follow on the same line.
    figure_start_re = re.compile(
        r"^\s*(figure|fig\.?)\s+\d{1,3}[A-Za-z]?\b(?![),;\]])\s*[.:\-]?\s+\S",
        re.IGNORECASE,
    )
    table_start_re = re.compile(
        r"^\s*(table|tab\.?)\s+\d{1,3}[A-Za-z]?\b(?![),;\]])\s*[.:\-]?\s+\S",
        re.IGNORECASE,
    )

    stop_heading_re = re.compile(
        r"^\s*(abstract|introduction|methods?|method|results?|discussion|conclusions?|references|bibliography|funding|acknowledg(?:e)?ments?)\s*$",
        re.IGNORECASE,
    )

    # Maximum lines to absorb after a caption start (typical captions are 1-6
    # wrapped lines; we cap to avoid swallowing whole paragraphs).
    _MAX_CONTINUATION_LINES = 6
    _MAX_CAPTION_CHARS = 1200

    def is_caption_start(line: str) -> bool:
        s = line.strip()
        return bool(figure_start_re.match(s) or table_start_re.match(s))

    def is_figure_start(line: str) -> bool:
        return bool(figure_start_re.match(line.strip()))

    def is_table_start(line: str) -> bool:
        return bool(table_start_re.match(line.strip()))

    def looks_like_body_prose(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        # Long line ending with a sentence terminator is almost always body prose.
        if len(s) > 140 and s[-1] in ".!?":
            return True
        # Lines that start with a known section-ish keyword.
        if re.match(
            r"^(however|moreover|therefore|in (?:contrast|addition|particular)|"
            r"we (?:found|observed|tested|report)|in the|among|because|"
            r"although|conclusion|background|results?|methods?|discussion)\b",
            s,
            re.IGNORECASE,
        ):
            return True
        return False

    def is_caption_continuation(line: str, current_caption_chars: int) -> bool:
        s = line.strip()
        if not s:
            return False
        if stop_heading_re.match(s):
            return False
        if is_caption_start(s):
            return False
        if looks_like_body_prose(s):
            return False
        if current_caption_chars + len(s) > _MAX_CAPTION_CHARS:
            return False
        if s[0].islower():
            return True
        if s[0] in "([":
            return True
        if re.match(r"^\d+\s*[\].:,\-]\s*", s):
            return True
        return False

    keep = [True] * len(lines)
    figures: list[str] = []
    tables: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if not is_caption_start(line):
            i += 1
            continue

        buf = [line.strip()]
        keep[i] = False
        cap_is_figure = is_figure_start(line)
        cap_is_table = is_table_start(line)

        j = i + 1
        continuation_count = 0
        while j < len(lines) and continuation_count < _MAX_CONTINUATION_LINES:
            nxt = lines[j]
            if not nxt.strip():
                break
            if stop_heading_re.match(nxt.strip()):
                break
            if is_caption_start(nxt):
                break
            caption_chars = sum(len(b) for b in buf)
            if not is_caption_continuation(nxt, caption_chars):
                break
            buf.append(nxt.strip())
            keep[j] = False
            j += 1
            continuation_count += 1

        caption = " ".join(buf).strip()
        # Skip captions that ended up obviously broken (single short fragment with
        # no content beyond the label) — they're usually inline references like
        # "(Fig. 4B)" that survived prose.
        if len(caption) < 20:
            i = j
            continue
        if cap_is_figure:
            figures.append(caption)
        elif cap_is_table:
            tables.append(caption)

        i = j

    pruned_lines = [ln for k, ln in zip(keep, lines) if k]
    pruned_text = "\n".join(pruned_lines).strip()
    return figures, tables, pruned_text


def _build_llm_text(record: dict[str, Any]) -> str:
    """
    Build the plain-text version for downstream LLMs: underlined section headers
    and body text only (no instructional signpost lines).

    References are intentionally omitted from this export — the LLM does not need
    the bibliography for screening. The raw reference list is still preserved in
    the `_full.txt` sidecar for inspection.
    """
    title, sections = build_llm_ready_payload(record)
    figures = record.get("figure_captions") or []
    tables = record.get("table_captions") or []

    def _underline(header: str, ch: str = "-") -> str:
        return ch * max(3, len(header))

    def _format_header(header: str) -> str:
        return f"{header}\n{_underline(header, ch='=')}"

    parts: list[str] = []
    if title:
        parts.append(_format_header("TITLE") + "\n" + title.strip())

    cite = (record.get("metadata") or {}).get("citation_label") or ""
    if cite.strip():
        parts.append(_format_header("CITATION") + "\n" + cite.strip())

    mapping = [
        ("ABSTRACT", "abstract"),
        ("INTRODUCTION", "introduction"),
        ("METHODS", "methods"),
        ("RESULTS", "results"),
        ("DISCUSSION", "discussion"),
    ]
    for header, key in mapping:
        content = (sections.get(key) or "").strip()
        if content:
            parts.append(_format_header(header) + "\n\n" + content)

    has_section_body = any(
        (sections.get(k) or "").strip()
        for k in ("abstract", "introduction", "methods", "results", "discussion")
    )
    if not has_section_body and record.get("clean_text"):
        body = strip_reference_tail_for_llm(record["clean_text"].strip())
        body = filter_llm_section_body(body)
        parts.append(_format_header("FULL TEXT") + "\n" + body)

    if figures:
        cap_lines = "\n".join(f"- {c.strip()}" for c in figures)
        parts.append(_format_header("FIGURE CAPTIONS") + "\n" + cap_lines)
    if tables:
        cap_lines = "\n".join(f"- {c.strip()}" for c in tables)
        parts.append(_format_header("TABLE CAPTIONS") + "\n" + cap_lines)

    out = "\n\n".join(parts).strip()
    out = strip_reference_tail_for_llm(out)
    out = normalize_llm_text_spacing(out)
    return out


def _build_record(
    source_path: str,
    clean_text: str,
    scanned_warning: str | None,
    pdf_metadata: dict[str, Any] | None = None,
    citation_year_suffix: str = "",
) -> dict[str, Any]:
    figure_captions, table_captions, pruned_text = _extract_captions_and_prune(clean_text)
    pruned_text = strip_reference_tail_for_llm(pruned_text)
    meta = extract_metadata_light(pruned_text)
    meta["title"] = refine_llm_title(pruned_text, meta.get("title") or "")
    if meta.get("year") is None:
        meta["year"] = _year_from_pdf_metadata(pdf_metadata)
    merge_authors_into_metadata(meta, pruned_text, pdf_metadata)
    meta["citation_label"] = build_metadata_citation_label(meta, citation_year_suffix)
    sections = segment_sections(pruned_text)
    record: dict[str, Any] = {
        "source_file": source_path,
        "metadata": meta,
        "sections": sections,
        "clean_text": pruned_text,
        "clean_text_full": clean_text,
        "figure_captions": figure_captions,
        "table_captions": table_captions,
        "chunks": chunk_text(pruned_text),
        "citation_year_suffix": citation_year_suffix,
    }
    if scanned_warning:
        record["warnings"] = [scanned_warning]

    record["llm_text"] = _build_llm_text(record)
    return record


def process_pdf(
    pdf_path: str | Path,
    out_dir: Path | None = None,
    write_sidecar: bool = True,
    simple_name: bool = False,
) -> dict[str, Any]:
    """
    Process one PDF to structured dict; optionally write LLM-ready TXT to out_dir.
    """
    path = Path(pdf_path).resolve()
    doc, raw = extract_full_text(str(path))
    pdf_meta: dict[str, Any] = {}
    try:
        scanned = is_likely_scanned(doc)
        if getattr(doc, "metadata", None):
            pdf_meta = dict(doc.metadata)
    finally:
        doc.close()

    warn = None
    if scanned:
        warn = (
            "Low text density per page: this PDF may be image-only. "
            "Install Tesseract and enable OCR in a future version for full recall."
        )

    cleaned_full = clean_full_text(raw)
    rec = _build_record(str(path), cleaned_full, warn, pdf_metadata=pdf_meta, citation_year_suffix="")

    if write_sidecar and out_dir is not None:
        out_dir = Path(out_dir)
        stem = path.stem
        jid = str(uuid.uuid4())[:8]
        base = out_dir / (stem if simple_name else f"{stem}_{jid}")
        txt_path = base.parent / f"{base.name}.txt"
        write_txt(txt_path, rec["llm_text"])

    return rec


def process_paths(
    pdf_paths: list[str | Path],
    out_dir: Path,
    zip_name: str | None = None,
) -> list[dict[str, Any]]:
    """Batch process; optional single ZIP of all TXT outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    paths_resolved: list[Path] = []
    zip_members: list[tuple[Path, str]] = []

    for p in pdf_paths:
        path = Path(p).resolve()
        paths_resolved.append(path)
        rec = process_pdf(path, out_dir=None, write_sidecar=False)
        results.append(rec)

    assign_citation_suffixes(results)
    for rec in results:
        suf = rec.get("citation_year_suffix") or ""
        rec["metadata"]["citation_label"] = build_metadata_citation_label(rec["metadata"], suf)
        rec["citation_year_suffix"] = suf
        rec["llm_text"] = _build_llm_text(rec)

    for path, rec in zip(paths_resolved, results):
        stem = path.stem
        jid = str(uuid.uuid4())[:8]
        tpath = out_dir / f"{stem}_{jid}.txt"
        write_txt(tpath, rec["llm_text"])
        zip_members.append((tpath, tpath.name))

    if zip_name and zip_members:
        zip_outputs(zip_members, out_dir / zip_name)

    return results
