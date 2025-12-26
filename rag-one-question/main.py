import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


# ----------------------------
# Utilities
# ----------------------------
def read_markdown_files(folder: str) -> List[Tuple[str, str]]:
    """Return list of (doc_id, text). doc_id is filename."""
    docs = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".md"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs.append((fname, f.read()))
    if not docs:
        raise RuntimeError(f"No .md files found in {folder}")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple character-based chunking with overlap.
    This is intentionally naive to demonstrate chunking sensitivity.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Create normalized embeddings so cosine similarity is a dot product.
    """
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def top_k_cosine(query_emb: np.ndarray, chunk_embs: np.ndarray, k: int) -> List[int]:
    """
    query_emb: (d,)
    chunk_embs: (n, d)
    returns indices of top-k most similar chunks.
    """
    sims = chunk_embs @ query_emb  # dot product since normalized -> cosine
    k = min(k, len(sims))
    top_idx = np.argsort(-sims)[:k]
    return top_idx.tolist()


# ----------------------------
# Mini-RAG pipeline
# ----------------------------
def build_chunks(data_folder: str, chunk_size: int, overlap: int) -> List[Chunk]:
    docs = read_markdown_files(data_folder)
    all_chunks: List[Chunk] = []
    for doc_id, text in docs:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(chunks):
            all_chunks.append(Chunk(doc_id=doc_id, chunk_id=i, text=ch))
    return all_chunks


def retrieve(model: SentenceTransformer, chunks: List[Chunk], question: str, top_k: int):
    chunk_texts = [c.text for c in chunks]
    chunk_embs = embed_texts(model, chunk_texts)
    q_emb = embed_texts(model, [question])[0]

    idxs = top_k_cosine(q_emb, chunk_embs, k=top_k)
    # compute scores for printing
    scores = (chunk_embs @ q_emb).tolist()

    results = []
    for rank, idx in enumerate(idxs, start=1):
        c = chunks[idx]
        results.append((rank, scores[idx], c))
    return results




def main():
    parser = argparse.ArgumentParser(description="Mini-RAG: 3 fixed questions, no chat, no memory.")

    parser.add_argument("--data-folder", default="data", help="Folder with .md files")
    parser.add_argument("--chunk-size", type=int, default=220, help="Chunk size (characters)")
    parser.add_argument("--overlap", type=int, default=40, help="Overlap between chunks (characters)")
    parser.add_argument("--top-k", type=int, default=3, help="How many chunks to retrieve")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model name")
    parser.add_argument("--debug", action="store_true", help="Print all chunks with IDs")
    args = parser.parse_args()

    # 1) Build chunks
    chunks = build_chunks(args.data_folder, chunk_size=args.chunk_size, overlap=args.overlap)

    print("\n=== Mini-RAG: Setup ===")
    print(f"Docs folder: {args.data_folder}")
    print(f"Chunks: {len(chunks)} | chunk_size={args.chunk_size} overlap={args.overlap}")
    print(f"Embedding model: {args.model}")

    if args.debug:
        print("\n--- All chunks ---")
        for c in chunks:
            print(f"[{c.doc_id} | chunk {c.chunk_id}] {c.text[:140]}{'...' if len(c.text) > 140 else ''}")

    # 2) Fixed questions (exactly 3)
    questions = [
        "Q1: What is the main advantage of separating content creation from formatting in OneLatex?",
        "Q2: How does OneLatex interpret text highlighted in green in OneNote?",
        #Alternative"Q2: Green-highlighted text in OneNote is interpreted as what in OneLatex?",
        "Q3: How does OneLatex interpret text highlighted in yellow in OneNote?"
    ]

    model = SentenceTransformer(args.model)

    print("\n=== Questions & Retrieval ===")
    for q in questions:
        print("\n" + "=" * 80)
        print(f"Q: {q}")

        results = retrieve(model, chunks, q, top_k=args.top_k)

        print("\nTop retrieved chunks:")
        for rank, score, c in results:
            preview = c.text.replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            print(f"{rank:>2}. score={score:.3f} | source={c.doc_id} | chunk={c.chunk_id}")
            print(f"    {preview}")

        # 3) Minimal "answer": use the best chunk as grounded output
        best = results[0][2]
        print("\nGrounded answer (minimal):")
        print(f"Source: {best.doc_id} (chunk {best.chunk_id})")
        print(best.text)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
