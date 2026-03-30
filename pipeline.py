#!/usr/bin/env python3
"""
CLI for full-text PDF ingestion (systematic review automation).

Reads academic PDFs, extracts and cleans text, segments common sections,
and emits JSON plus plain-text exports (see README).

Examples:
  python pipeline.py article.pdf -o output
  python pipeline.py papers/*.pdf -o output --zip bundle.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as script from project root
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sr_pipeline.pipeline import process_paths, process_pdf  # noqa: E402


def _configure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows consoles (cp1252) when printing titles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def main() -> int:
    _configure_stdio_utf8()
    ap = argparse.ArgumentParser(description="PDF → structured text for systematic reviews")
    ap.add_argument(
        "pdfs",
        nargs="+",
        help="One or more PDF files",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for JSON and TXT (default: ./output)",
    )
    ap.add_argument(
        "--zip",
        metavar="NAME",
        help="Also write a ZIP of all outputs inside the output directory",
    )
    ap.add_argument(
        "--single",
        action="store_true",
        help="Only first PDF; simpler filenames without batch UUID suffix",
    )
    args = ap.parse_args()

    pdfs = [Path(p).resolve() for p in args.pdfs]
    for p in pdfs:
        if not p.is_file():
            print(f"Not found: {p}", file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.single or len(pdfs) == 1:
        rec = process_pdf(
            pdfs[0],
            out_dir=args.output_dir,
            write_sidecar=True,
            simple_name=bool(args.single or len(pdfs) == 1),
        )
        if rec.get("warnings"):
            for w in rec["warnings"]:
                print(w, file=sys.stderr)
        cite = (rec.get("metadata") or {}).get("citation_label")
        title = rec["metadata"].get("title") or pdfs[0].name
        if cite:
            print(cite)
        print(title)
        return 0

    process_paths(pdfs, args.output_dir, zip_name=args.zip)
    print(f"Wrote {len(pdfs)} paper bundle(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
