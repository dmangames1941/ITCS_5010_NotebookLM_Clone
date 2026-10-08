import os
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import poml

from src.vectorstore import VectorStoreManager
from src.notebook_manager import NotebookManager

load_dotenv()

class ArtifactGenerator:
    def __init__(self, vector_store: VectorStoreManager, notebook_mgr: NotebookManager, model_name: str = "openai/gpt-oss-120b"):
        self.vector_store = vector_store
        self.notebook_mgr = notebook_mgr

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Check your .env file or Hugging Face secrets.")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.3
        )

    def _gather_notebook_context(self, notebook_id: str, max_chunks: int = 15) -> str:
        """Retrieves representative context chunks across notebook sources."""

        chunks = self.vector_store.similarity_search(notebook_id, query="overview main summary key points core topics", top_k=max_chunks)

        if not chunks:
            return ""

        context_str = ""
        for i, chunk in enumerate(chunks):
            src = chunk["metadata"].get("source_name", "Source")
            context_str += f"\n--- Source Document: {src} ---\n{chunk['text']}\n"

        return context_str

    def generate_report(self, notebook_id: str) -> str:
        context = self._gather_notebook_context(notebook_id)
        if not context:
            return "No documents available in this notebook to generate a report."

        poml_template = """
        <poml>
            <role>
                You are an expert technical editor and research analyst.
            </role>
            <task>
                1. Analyze the provided notebook source material below.
                2. Write a detailed, highly structure Executive Report in Markdown format.
                3. Structure the report with the following sections:
                    - # Executive Summary
                    - ## Key Topics and Themes
                    - ## Detailed Analysis
                    - ## Critical Findings and Takeaways
                    - ## Conclusion
                4. Use clear bullet points, clean headings, and inline bolding for key terms.
            </task>
        </poml>
        """
        base_instructions = poml.poml(poml_template)
        if isinstance(base_instructions, dict):
            base_instructions = base_instructions.get("content", str(base_instructions))
        else:
            base_instructions = str(base_instructions)
        final_prompt = f"{base_instructions}\n\nContext Documents:\n{context}"
        response = self.llm.invoke(final_prompt)
        return response.content if hasattr(response, "content") else str(response)

    def generate_quiz(self, notebook_id: str) -> str:
        # Fix method call here as well
        context = self._gather_notebook_context(notebook_id)
        if not context:
            return "No documents available in this notebook to generate a quiz."

        poml_template = """<poml>
        <role>
            You are an expert educator and exam designer.
        </role>
        <task>
            1. Analyze the provided notebook source material below.
            2. Create a comprehensive 5-question multiple-choice quiz based ONLY on the context.
            3. For each question:
                - Provide 4 options (A, B, C, D).
                - Clearly state the correct answer.
                - Provide a brief explanation citing the context.
        </task>
    </poml>"""

        base_instructions = poml.poml(poml_template)
        if isinstance(base_instructions, dict):
            base_instructions = base_instructions.get("content", str(base_instructions))
        else:
            base_instructions = str(base_instructions)

        final_prompt = f"{base_instructions}\n\nContext Documents:\n{context}"

        response = self.llm.invoke(final_prompt)
        return response.content if hasattr(response, "content") else str(response)