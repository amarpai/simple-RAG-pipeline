"""A simple RAG (retrieval-augmented generation) over one PDF.

Pipeline:
  1. index  : PDF -> text per page -> chunks -> embeddings (+ BM25 keyword index) -> saved to disk
  2. ask    : question -> hybrid search (vector + keyword, fused) -> top chunks -> Claude answers with page citations

Usage:
  python rag.py index path/to/file.pdf
  python rag.py search "your question"      # retrieval only, no API call (free)
  python rag.py ask "your question"         # retrieval + Claude answer
  python rag.py chat                        # interactive loop
"""

import json
import os
import re
import sys
from pathlib import Path

import anthropic
import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small local embedding model (~130 MB, runs on CPU)
CLAUDE_MODEL = "claude-opus-5-5"
CHUNK_CHARS = 800      # target chunk size
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks
TOP_K = 6              # chunks handed to Claude

_embedder = None


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ---------- 1. Indexing ----------

def chunk_page(text: str) -> list[str]:
    """Split one page into overlapping chunks. Short pages (most slides) stay whole."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        # try to break at a newline or sentence end instead of mid-word
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + CHUNK_CHARS // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def build_index(pdf_path: str) -> None:
    reader = PdfReader(pdf_path)
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        for piece in chunk_page(page.extract_text() or ""):
            if len(piece) > 20:  # skip near-empty slides
                chunks.append({"page": page_no, "text": piece})

    print(f"{len(reader.pages)} pages -> {len(chunks)} chunks. Embedding...")
    # bge models embed passages as-is; queries get an instruction prefix (see query_embed)
    vectors = np.array(list(embedder().passage_embed([c["text"] for c in chunks])), dtype=np.float32)

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps({"source": str(pdf_path), "chunks": chunks}, ensure_ascii=False, indent=1)
    )
    print(f"Saved index to {INDEX_DIR}/")


# ---------- 2. Retrieval ----------

def load_index():
    if not (INDEX_DIR / "chunks.json").exists():
        sys.exit("No index yet. Run: python rag.py index <file.pdf>")
    data = json.loads((INDEX_DIR / "chunks.json").read_text())
    vectors = np.load(INDEX_DIR / "vectors.npy")
    return data["source"], data["chunks"], vectors


def search(question: str, k: int = TOP_K) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keywords, merged with Reciprocal Rank Fusion."""
    _, chunks, vectors = load_index()

    # semantic (meaning) ranking
    q_vec = np.array(next(iter(embedder().query_embed(question))), dtype=np.float32)
    sims = vectors @ q_vec / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec))
    vec_rank = np.argsort(-sims)

    # keyword ranking (catches exact terms, names, acronyms like "CSR")
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    kw_rank = np.argsort(-bm25.get_scores(tokenize(question)))

    # Reciprocal Rank Fusion: score = sum of 1 / (60 + rank) across both lists
    fused = {}
    for ranking in (vec_rank, kw_rank):
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank)

    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**chunks[i], "score": fused[i], "similarity": float(sims[i])} for i in best]


# ---------- 3. Generation ----------

SYSTEM_PROMPT = """You answer questions about a course document using only the excerpts provided.
Each excerpt is tagged with its page number. Cite pages inline like (p. 12).
If the excerpts don't contain the answer, say so plainly instead of guessing; you may then add
general knowledge, clearly labelled as not coming from the document."""


def answer(question: str, client: anthropic.Anthropic) -> str:
    hits = search(question)
    context = "\n\n".join(f"<excerpt page=\"{h['page']}\">\n{h['text']}\n</excerpt>" for h in hits)

    try:
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            # if the model declines on safety grounds, the API retries on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{
                "role": "user",
                "content": f"<document_excerpts>\n{context}\n</document_excerpts>\n\nQuestion: {question}",
            }],
        )
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed: set ANTHROPIC_API_KEY (see README.md).")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the API; wait a moment and retry.")
    except anthropic.APIStatusError as e:
        sys.exit(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("Network error: could not reach the Anthropic API.")

    if response.stop_reason == "refusal":
        return "(The model declined to answer this question.)"
    text = "".join(b.text for b in response.content if b.type == "text")
    pages = sorted({h["page"] for h in hits})
    return f"{text}\n\n[retrieved pages: {', '.join(map(str, pages))}]"


# ---------- CLI ----------

def make_client() -> anthropic.Anthropic:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("No Anthropic credentials found. Run: export ANTHROPIC_API_KEY=sk-ant-...  (see README.md)")
    return anthropic.Anthropic()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]

    if cmd == "index" and len(sys.argv) == 3:
        build_index(sys.argv[2])
    elif cmd == "search" and len(sys.argv) >= 3:
        for h in search(" ".join(sys.argv[2:])):
            print(f"\n--- page {h['page']}  (fused {h['score']:.4f}, cosine {h['similarity']:.3f})")
            print(h["text"][:400])
    elif cmd == "ask" and len(sys.argv) >= 3:
        print(answer(" ".join(sys.argv[2:]), make_client()))
    elif cmd == "chat":
        client = make_client()
        print("Ask about the PDF (empty line to quit).")
        while q := input("\n> ").strip():
            print("\n" + answer(q, client))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
