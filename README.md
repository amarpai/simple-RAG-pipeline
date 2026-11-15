# PDF RAG

Ask questions about a PDF. Retrieval runs on your Mac; Claude writes the answer and cites page numbers.

## Setup (already done)
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

## Use
    export ANTHROPIC_API_KEY=sk-ant-...        # from https://console.anthropic.com
    .venv/bin/python rag.py index "/path/to/file.pdf"   # rebuild for a new PDF
    .venv/bin/python rag.py search "question"           # see what retrieval finds (free)
    .venv/bin/python rag.py ask "question"              # full RAG answer
    .venv/bin/python rag.py chat                        # interactive

## How it works
1. **Extract + chunk**: pypdf text per page, split into ~800-char chunks with overlap, page number kept.
2. **Embed**: local `bge-small-en-v1.5` model turns each chunk into a vector (saved in `index/`).
3. **Hybrid search**: cosine similarity (meaning) + BM25 (exact keywords), merged with Reciprocal Rank Fusion.
4. **Generate**: top 6 chunks go to `claude-opus-5-5` with instructions to answer only from them and cite pages.
