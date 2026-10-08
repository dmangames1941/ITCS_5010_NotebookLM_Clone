---
title: NotebookLM Clone
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
---

# NotebookLM Clone

An interactive AI-powered research assistant inspired by Google's NotebookLM. This application allows users to upload custom documents (PDFs, PowerPoint presentations, web URLs, and plain text) to build a personalized vector-indexed knowledge base, generate synthetic artifacts (study guides, executive summaries, FAQs, timeline overviews), and interact with their sources via context-aware RAG (Retrieval-Augmented Generation) chat.

## Key Features

* **Multi-Format Document Ingestion**: Upload and parse PDFs (`.pdf`), PowerPoint presentations (`.pptx`), plain text files (`.txt`), or fetch live web pages via URL.
* **Smart Vector Storage**: Uses `ChromaDB` and local Hugging Face sentence transformers (`sentence-transformers/all-MiniLM-L6-v2`) to chunk and embed documents for fast semantic retrieval.
* **Context-Aware RAG Chat**: Chat with your documents using Groq-powered LLMs (`llama-3.3-70b-versatile`) with direct source citations.
* **Automated Artifact Generation**: Automatically synthesize study guides, executive summaries, FAQs, and structured timelines from selected sources.
* **ZeroGPU & Hugging Face Spaces Ready**: Optimized for deployment on Hugging Face Spaces with dynamic GPU allocation using `@spaces.GPU`.

## Application Structure

```text
.
├── app.py                  # Main Gradio web application UI and event handlers
├── requirements.txt        # Python package dependencies
├── README.md               # Space configuration & project documentation
└── src/
    ├── document_loader.py  # Utility functions to parse PDFs, PPTXs, TXTs, and Web URLs
    ├── vectorstore.py      # ChromaDB setup, text splitting, and embedding pipelines
    ├── rag_chain.py        # LangChain & Groq LLM chains for chat and document synthesis
    └── utils.py            # Helper functions and formatting utilities
```

## How Data is Stored
* In-Memory & Ephemeral Storage: Documents uploaded during a session are stored in an isolated, temporary session directory (./temp_uploads) and indexed into an ephemeral or local file-backed ChromaDB vector database instance (./chroma_db).
* Embeddings: Text chunks are converted into 384-dimensional dense vector embeddings using the sentence-transformers/all-MiniLM-L6-v2 model.
* Privacy & Isolation: No raw documents or vector databases are uploaded to external third-party vector hosts. Embeddings are computed locally and queries are transmitted securely via API to Groq for text generation.

## Required Environment Variables
To run the application locally or on Hugging Face Spaces, you must configure the following environment variable:
* GROQ_API_KEY: API key from Groq Console used for fast LLM inference (openai/gpt-oss-120b).

When deploying to Hugging Face Spaces, add GROQ_API_KEY under Settings -> Variables and secrets.

## How to Run Locally

### Prerequisites
* Python 3.10 or higher
* Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dmangames1941/ITCS_5010_NotebookLM_Clone](https://github.com/dmangames1941/ITCS_5010_NotebookLM_Clone)
   cd ITCS_5010_NotebookLM_Clone
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Set up Environment Variables:**
Create a .env file in the root directory:
* Linux / macOS:
   ```bash
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
* Windows (Command Prompt / CMD):
   ```bash
   echo GROQ_API_KEY=your_groq_api_key_here > .env
   ```
* Windows (PowerShell):
   ```PowerShell
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
5. **Run the Gradio Application**
   ```bash
   python app.py
   ```
   Open your browser and navigate to http://localhost:7860.
