"""RAG retrieval engine for the University IT policies knowledge base."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from policy_loader import load_policy_documents


class PolicyRAG:
    """Single-document policy RAG retriever."""

    def __init__(
        self,
        embed_model_name: str = "models/qwen3-embedding-0.6b",
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f">>> RAG: loading embedding model from {embed_model_name} ...", flush=True)
        self.embed_model = SentenceTransformer(
            embed_model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(">>> RAG: embedding model loaded.", flush=True)

        self.documents = load_policy_documents()
        self.chunks = []
        self.metadatas = []
        self.index = None

        self._build_index()

    def _chunk_document(self, text: str):
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _build_index(self):
        print(">>> RAG: chunking policy documents ...", flush=True)
        for doc in self.documents:
            sub_chunks = self._chunk_document(doc.page_content)
            for chunk in sub_chunks:
                self.chunks.append(chunk)
                self.metadatas.append({
                    "policy_id": doc.metadata["policy_id"],
                    "topic": doc.metadata["topic"],
                    "source": doc.metadata["source"],
                })

        if not self.chunks:
            raise ValueError("No chunks produced from policy documents.")

        print(f">>> RAG: encoding {len(self.chunks)} chunks (CPU, may take a few minutes) ...", flush=True)
        embeddings = self.embed_model.encode(
            self.chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f">>> RAG: indexed {len(self.chunks)} chunks (dim={dim}).", flush=True)

    def retrieve(self, query: str, top_k: int = 4, score_threshold: float = 0.35):
        if self.index is None:
            raise RuntimeError("Index not built.")

        query_vec = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_vec = np.array(query_vec).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < score_threshold:
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "policy_id": self.metadatas[idx]["policy_id"],
                "topic": self.metadatas[idx]["topic"],
            })
        return results

    def get_context(self, query: str, top_k: int = 4) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""
        return "\n\n---\n\n".join(r["text"] for r in results)