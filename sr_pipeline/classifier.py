"""Detect native text vs likely scanned (image-only) PDFs."""

from __future__ import annotations

import fitz


def is_likely_scanned(doc: fitz.Document, min_chars_per_page: int = 50) -> bool:
    """
    Heuristic: if average printable text per page is very low, assume scan/OCR needed.
    """
    n = doc.page_count
    if n == 0:
        return True
    total = 0
    for i in range(n):
        total += len(doc.load_page(i).get_text("text") or "")
    avg = total / n
    return avg < min_chars_per_page
