import os
import time
import html
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import poml

from src.vectorstore import VectorStoreManager

load_dotenv()

class RAGEngine:
    def __init__(self, vector_store: VectorStoreManager, model_name: str = "openai/gpt-oss-120b"):
        self.vector_store = vector_store

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Check your .env file or Hugging Face secrets.")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.2
        )

    def _build_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Builds the system instructions via POML, then safely appends raw context and query in Python.
        """
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk["metadata"].get("source_name", "Unknown Source")
            idx = chunk["metadata"].get("chunk_index", "N/A")
            context_str += f"\n--- Document [{i+1}]: {source} (Chunk {idx}) ---\n{chunk['text']}\n"

        poml_template = """<poml>
            <role>
                You are a precise, helpful AI research assistant for notebook source materials.
            </role>
            <task>
                1. Answer the question thoroughly based ONLY on the provided Context Documents below.
                2. If the context does not contain enough information to answer, state clearly that the sources do not contain the required information.
                3. ALWAYS cite your sources inline using the format.
            </task>
        </poml>"""

        base_instructions = poml.poml(poml_template)
        if isinstance(base_instructions, dict):
            base_instructions = base_instructions.get("content", str(base_instructions))
        else:
            base_instructions = str(base_instructions)

        final_prompt = f"{base_instructions}\n\nContext Documents:\n{context_str}\n\nUser Question:\n{query}"
        return final_prompt
    
    def retrieve_chunks(self, notebook_id: str, query:str, method: str = "vector", top_k: int = 4) -> Tuple[List[Dict[str, Any]], float]:
        """
        Retrieves context chunks using either Basic Vector Search or Reranked Search.

        Parameters:
            notebook_id: Active notebook ID
            query: Natural language user question
            method: 'vector' for Basic Similarity or 'rerank' for Two-Stage Reranking
            top_k: Final number of chunks to return
        Returns:
            Tuple of (Retrieved Cchunks, Retrieval Time in seconds)
        """
        start_time = time.time()
        if method == "rerank":
            raw_chunks = self.vector_store.similarity_search(notebook_id, query, top_k=10)
            query_terms = set(query.lower().split())
            for chunk in raw_chunks:
                text_terms = set(chunk["text"].lower().split())
                overlap = len(query_terms.intersection(text_terms))
                chunk["rerank_score"] = chunk["score"] + (overlap * 0.1)
            reranked_chunks = sorted(raw_chunks, key=lambda x: x["rerank_score"], reverse=True)
            final_chunks = reranked_chunks[:top_k]
        else:
            final_chunks = self.vector_store.similarity_search(notebook_id, query, top_k=top_k)

        retrieval_time = round(time.time() - start_time, 3)
        return final_chunks, retrieval_time

    def answer_question(self, notebook_id: str, query: str, retrieval_method: str = "vector", top_k: int = 4) -> Dict[str, Any]:
        """
            Executes full RAG workflow: Retrieves chunks, builds POML prompt, calls Groq, extracts citations.

        Parameters:
            notebook_id: Active notebook ID
            retrieval_method: 'vector' for Basic Similarity or 'rerank' for Two-Stage Reranking
            top_k: Number of chunks to retrieve and consider

        Returns:
            Dictionary containing the answer and citations
        """
        total_start = time.time()

        chunks, retrieval_time = self.retrieve_chunks(notebook_id, query, method=retrieval_method, top_k=top_k)

        if not chunks:
            return{
                "answer": "No sources found for this notebook. Please upload or link a source document first.",
                "citations": [],
                "retrieved_chunks": [],
                "retrieval_time": retrieval_time,
                "total_time": round(time.time() - total_start, 3),
                "method": retrieval_method
            }

        prompt_content = self._build_rag_prompt(query, chunks)
        response = self.llm.invoke(prompt_content)
        answer_text = response.content
        citations = []
        for chunk in chunks:
            src = chunk["metadata"].get("source_name", "Unknown")
            idx = chunk["metadata"].get("chunk_index", "N/A")
            citations.append(f"{src} (Chunk {idx})")

        total_time = round(time.time() - total_start, 3)
        return {
            "answer": answer_text,
            "citations": list(set(citations)),
            "retrieved_chunks": chunks,
            "retrieval_time": retrieval_time,
            "total_time": total_time,
            "method": retrieval_method
        }