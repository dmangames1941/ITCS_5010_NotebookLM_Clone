import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DATA_DIR = "data/chroma_db"

class VectorStoreManager:
    def __init__(self, persist_directory: str = CHROMA_DATA_DIR):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _get_collection_name(self, notebook_id: str) -> str:
        """Sanitizes notebook_id to comply with ChromaDB collection naming rules."""
        return f"notebook_{notebook_id.replace('-', '_')}"

    def get_or_create_collection(self, notebook_id: str):
        """Retrieves or creates a dedicated Chroma collection for a notebook."""
        collection_name = self._get_collection_name(notebook_id)
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, notebook_id: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds and stores document chunks in the notebook's collection.

        Params:
            notebook_id: Owning notebook ID
            chunks: List of dicts containing 'text' and 'metadata'
        """
        if not chunks:
            return

        collection = self.get_or_create_collection(notebook_id)
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        ids = [
            f"{chunk['metadata']['source_name']}_chunk_{chunk['metadata']['chunk_index']}_{i}"
            for i, chunk in enumerate(chunks)
        ]

        embeddings = self.embedding_model.embed_documents(texts)

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def similarity_search(self, notebook_id: str, query:str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search for a given notebook query.
        Params
            notebook_id: ID of the active notebook
            query: Natural language user prompt
            top_k: Number of relebant chunks to retrieve
        Returns
            List of retrieved chunks with text, metadata, and distance score
        """
        collection = self.get_or_create_collection(notebook_id)

        if collection.count() == 0:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        if results and results.get("documents"):
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                retrieved_chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "score": round(1.0 - dist, 4)
                })
        return retrieved_chunks

    def delete_notebook_collection(self, notebook_id: str) -> bool:
        """Completely deletes a notebook's vector collection when the notebook is removed."""
        collection_name = self._get_collection_name(notebook_id)
        try:
            self.client.delete_collection(name=collection_name)
            return True
        except ValueError:
            return False