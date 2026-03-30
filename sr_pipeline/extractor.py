"""Extract full text from PDFs: pdfplumber first (layout + tables), PyMuPDF fallback."""

from __future__ import annotations

import os
from typing import Any

import fitz

from sr_pipeline.cleaner import (
    clean_page_raw_text,
    estimate_reversed_line_ratio,
    strip_repeating_head_lines_from_pages,
    strip_repeating_tail_lines_from_pages,
)

# Minimum characters to treat pdfplumber output as usable for a page
_MIN_PAGE_CHARS = 24
# If this fraction of lines look like mirrored Latin, prefer PyMuPDF for the page
_REVERSED_LINE_RATIO_FALLBACK = 0.06


def _lines_from_words(words: list[dict[str, Any]], y_tol: float = 3.0) -> str:
    """Join word dicts from pdfplumber into lines (top-to-bottom, left-to-right)."""
    if not words:
        return ""
    words_sorted = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))
    lines: list[str] = []
    current: list[dict[str, Any]] = []
    last_top: float | None = None
    for w in words_sorted:
        top = float(w["top"])
        if last_top is None:
            last_top = top
            current.append(w)
        elif abs(top - last_top) <= y_tol:
            current.append(w)
        else:
            current.sort(key=lambda x: float(x["x0"]))
            lines.append(" ".join(x["text"] for x in current))
            current = [w]
            last_top = top
    if current:
        current.sort(key=lambda x: float(x["x0"]))
        lines.append(" ".join(x["text"] for x in current))
    return "\n".join(lines)


def _two_column_lines(page: Any) -> str | None:
    """
    If the page looks like two text columns, read left column then right.
    Otherwise return None and let single-column logic handle it.
    """
    words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    if not words or len(words) < 80:
        return None
    w_px = float(page.width)
    if w_px <= 0:
        return None
    xs = [float(w["x0"]) for w in words]
    span = max(xs) - min(xs)
    if span < 0.55 * w_px:
        return None
    mid = 0.52 * w_px
    left = [w for w in words if float(w["x0"]) < mid]
    right = [w for w in words if float(w["x0"]) >= mid]
    if len(left) < 50 or len(right) < 50:
        return None
    return _lines_from_words(left) + "\n\n" + _lines_from_words(right)


def _format_tables(page: Any) -> str:
    """Append detected tables with a clear delimiter (helps LLMs find tabular data)."""
    try:
        tables = page.extract_tables() or []
    except Exception:
        return ""
    blocks: list[str] = []
    for table in tables:
        if not table:
            continue
        rows_out: list[str] = []
        for row in table:
            cells = [str(c or "").replace("\n", " ").strip() for c in row]
            if not any(cells):
                continue
            rows_out.append(" | ".join(cells))
        if len(rows_out) >= 2:
            blocks.append("[Table]\n" + "\n".join(rows_out))
    if not blocks:
        return ""
    return "\n\n" + "\n\n".join(blocks)


def _extract_page_pdfplumber(page: Any) -> str:
    """
    Best-effort text for one pdfplumber page: layout text, then plain, then words.
    Tables are appended when extract_tables finds structured grids.
    """
    parts: list[str] = []
    t_layout = (page.extract_text(layout=True, x_tolerance=3, y_tolerance=3) or "").strip()
    if t_layout:
        parts.append(t_layout)
    else:
        t_plain = (page.extract_text(x_tolerance=3, y_tolerance=3) or "").strip()
        if t_plain:
            parts.append(t_plain)

    body = "\n\n".join(parts) if parts else ""

    if len(body) < _MIN_PAGE_CHARS:
        col = _two_column_lines(page)
        if col:
            body = col
        else:
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            wtxt = _lines_from_words(words).strip()
            if len(wtxt) > len(body):
                body = wtxt

    body = body.strip()
    if body:
        body = body + _format_tables(page)
    return body


def _pymupdf_page_text(doc: fitz.Document, index: int) -> str:
    return doc.load_page(index).get_text("text") or ""


def extract_full_text(pdf_path: str) -> tuple[fitz.Document, str]:
    """
    Concatenate page text in reading order.

    Uses pdfplumber (layout + optional tables) when available; falls back to
    PyMuPDF for a page when pdfplumber yields little or no text. The returned
    ``fitz.Document`` is used by the scanned-PDF heuristic and should be closed
    by the caller.
    """
    doc = fitz.open(pdf_path)
    n = doc.page_count
    raw_pages: list[str] = [""] * n

    use_plumber = os.environ.get("SR_PDF_BACKEND", "auto").lower() != "pymupdf"

    plumber_pdf = None
    if use_plumber:
        try:
            import pdfplumber

            plumber_pdf = pdfplumber.open(pdf_path)
        except Exception:
            plumber_pdf = None

    try:
        if plumber_pdf is not None and len(plumber_pdf.pages) == n:
            for i in range(n):
                ptxt = _extract_page_pdfplumber(plumber_pdf.pages[i]).strip()
                ftxt = _pymupdf_page_text(doc, i).strip()
                rev_ratio = estimate_reversed_line_ratio(ptxt)
                if rev_ratio > _REVERSED_LINE_RATIO_FALLBACK:
                    raw_pages[i] = ftxt or ptxt
                elif len(ptxt) < _MIN_PAGE_CHARS:
                    raw_pages[i] = ftxt or ptxt
                elif len(ptxt) < 500 and len(ftxt) > max(len(ptxt) * 2, len(ptxt) + 400):
                    raw_pages[i] = ftxt
                else:
                    raw_pages[i] = ptxt
        else:
            for i in range(n):
                raw_pages[i] = _pymupdf_page_text(doc, i)
    finally:
        if plumber_pdf is not None:
            try:
                plumber_pdf.close()
            except Exception:
                pass

    raw_pages = strip_repeating_head_lines_from_pages(raw_pages)
    raw_pages = strip_repeating_tail_lines_from_pages(raw_pages)

    parts: list[str] = []
    for i in range(n):
        parts.append(clean_page_raw_text(raw_pages[i], i + 1, n))
    return doc, "\n\n".join(parts)
