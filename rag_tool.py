"""RAG tool wrapper around PolicyRAG."""

from policy_loader import load_policy_documents
from rag import PolicyRAG


class RAGTool:
    """Wrap RAG retrieval into a simple search interface."""

    def __init__(self, embed_model_name: str = "models/qwen3-embedding-0.6b"):
        print(">>> RAGTool: initializing ...", flush=True)
        self.rag = PolicyRAG(embed_model_name=embed_model_name)
        self.documents = load_policy_documents()
        print(">>> RAGTool: ready.", flush=True)

    def search(self, query: str, top_k: int = 4) -> str:
        return self.rag.get_context(query, top_k=top_k)

    def search_with_meta(self, query: str, top_k: int = 4):
        return self.rag.retrieve(query, top_k=top_k)