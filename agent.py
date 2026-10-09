"""University IT Service Agent: RAG + qwen3-0.6b generation + anti-hallucination refusal."""

print(">>> agent.py: script started", flush=True)

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(">>> agent.py: torch & transformers imported", flush=True)

from rag_tool import RAGTool

print(">>> agent.py: rag_tool imported", flush=True)


REFUSAL_MESSAGE = (
    "I'm sorry, but I don't have that information. "
    "The university IT policies provided to me do not contain a verified "
    "Service Desk phone number. Please check the university's official "
    "IT Service Desk webpage or contact them through the official channel."
)


class ITServiceAgent:
    """University IT support agent powered by RAG + qwen3-0.6b."""

    def __init__(
        self,
        model_name: str = "models/qwen3-0.6b",
        embed_model_name: str = "models/qwen3-embedding-0.6b",
    ):
        print(">>> agent: loading tokenizer ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
        print(">>> agent: tokenizer loaded.", flush=True)

        print(">>> agent: loading generation model (this may take 30-60s) ...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            local_files_only=True,
        )
        self.model.eval()
        print(">>> agent: generation model loaded.", flush=True)

        print(">>> agent: building RAG ...", flush=True)
        self.rag_tool = RAGTool(embed_model_name=embed_model_name)
        print(">>> agent: RAG ready.", flush=True)

    def _build_prompt(self, query: str, context: str) -> str:
        prompt = (
            "You are a university IT support assistant. "
            "Answer the user's question using ONLY the context provided below.\n\n"
            "Rules:\n"
            "1. If the context contains the answer, answer concisely and accurately.\n"
            "2. If the context does NOT contain the answer, you MUST reply exactly with: "
            f"\"{REFUSAL_MESSAGE}\"\n"
            "3. Never invent phone numbers, email addresses, URLs, or any contact "
            "details that are not explicitly present in the context.\n"
            "4. Never provide instructions or advice that is not supported by the context.\n\n"
            f"Context:\n{context}\n\n"
            f"User: {query}\n\n"
            "Assistant:"
        )
        return prompt

    def _generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text

    @staticmethod
    def _contains_phone_number(text: str) -> bool:
        patterns = [
            r"\+?\d[\d\s\-\(\)]{6,}\d",
            r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
            r"\b\d{4}[-.\s]\d{4}\b",
        ]
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _context_has_phone(context: str) -> bool:
        patterns = [
            r"\+?\d[\d\s\-\(\)]{6,}\d",
            r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
        ]
        return any(re.search(p, context) for p in patterns)

    def answer(self, query: str) -> str:
        print(f">>> agent: retrieving context for query ...", flush=True)
        context = self.rag_tool.search(query, top_k=4)

        if not context.strip():
            print(">>> agent: no context retrieved -> refusing.", flush=True)
            return REFUSAL_MESSAGE

        print(">>> agent: building prompt ...", flush=True)
        prompt = self._build_prompt(query, context)

        print(">>> agent: generating response ...", flush=True)
        response = self._generate(prompt)
        print(">>> agent: generation done.", flush=True)

        if not response or len(response.strip()) < 3:
            return REFUSAL_MESSAGE

        if not self._context_has_phone(context) and self._contains_phone_number(response):
            return REFUSAL_MESSAGE

        if "don't have that information" in response.lower():
            return REFUSAL_MESSAGE

        return response


if __name__ == "__main__":
    print(">>> main: creating agent ...", flush=True)
    agent = ITServiceAgent()
    print(">>> main: agent created.", flush=True)

    test_query = (
        "What exact phone number should I call for the "
        "University IT Service Desk?"
    )

    print("=" * 70, flush=True)
    print(f"Query: {test_query}", flush=True)
    print("=" * 70, flush=True)

    answer = agent.answer(test_query)

    print(f"\nAnswer:\n{answer}", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("Expected: the model should refuse and not invent a phone number.", flush=True)