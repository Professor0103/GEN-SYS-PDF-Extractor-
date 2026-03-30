# GEN-SYS (PDF Full-Text Extractor)

Extract and normalize full text from academic PDFs for systematic review and LLM-assisted screening. The pipeline uses native PDF text (PyMuPDF / pdfplumber), reflows broken lines, strips common publisher noise, segments coarse sections (abstract, methods, results, etc.), and writes JSON plus a plain-text view for downstream tools.

**Scope:** Works best on born-digital PDFs with selectable text. Scanned pages are detected; OCR is not included here (add Tesseract + `pytesseract` if you need it).

## Requirements

- Python 3.10+
- Dependencies: see `requirements.txt` or `pyproject.toml`

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
```

## Usage

Single PDF (writes `stem.json`, `stem.txt`, `stem_full.txt` under the output directory):

```bash
python pipeline.py path/to/article.pdf -o output
```

Batch (UUID suffix on filenames; optional ZIP):

```bash
python pipeline.py papers/*.pdf -o output --zip results.zip
```

Outputs:

| File | Role |
|------|------|
| `*.json` | Metadata, sections, citation label, chunks |
| `*.txt` | LLM-oriented text (references list omitted by design) |
| `*_full.txt` | Cleaned full text before aggressive reference trimming |

Optional JSON configs at the repo root (`boilerplate_line_patterns.json`, `reflow_token_fixes.json`) extend line removal and token fixes without editing code.

## Layout

- `pipeline.py` — CLI
- `sr_pipeline/` — extract → clean → segment → export

## Citation

If you use this software in a publication, cite the repository (and the paper it accompanies) as you would any research artifact. Replace the placeholder in `LICENSE` with your institution or author line if you fork the project.

## License

MIT — see `LICENSE`.
