"""Systematic review PDF ingestion: extract, clean, segment, export."""

from sr_pipeline.pipeline import process_pdf, process_paths

__all__ = ["process_pdf", "process_paths"]
