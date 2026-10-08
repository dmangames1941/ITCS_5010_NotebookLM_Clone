import os
import gradio as gr
from dotenv import load_dotenv
import tempfile

from src.notebook_manager import NotebookManager
from src.ingestion import DocumentIngestor
from src.vectorstore import VectorStoreManager
from src.rag_engine import RAGEngine
from src.artifacts import ArtifactGenerator

load_dotenv()

notebook_mgr = NotebookManager()
ingestor = DocumentIngestor()
vector_store = VectorStoreManager()
rag_engine = RAGEngine(vector_store=vector_store)
artifact_gen = ArtifactGenerator(vector_store=vector_store, notebook_mgr=notebook_mgr)

def get_notebook_choices():
    """Fetches formatted dropdown choices for active notebooks."""
    notebooks = notebook_mgr.list_notebooks()
    if not notebooks:
        return []
    return [(nb["name"], nb["id"]) for nb in notebooks]

def initialize_app():
    """Initializes UI state upon page load."""
    choices = get_notebook_choices()
    if choices:
        first_id = choices[0][1]
        return gr.Dropdown(choices=choices, value=first_id), first_id
    return gr.Dropdown(choices=[], value=None), None

def handle_create_notebook(name):
    """Creates a new notebook and updates dropdown selection."""
    if not name or not name.strip():
        choices = get_notebook_choices()
        current_val = choices[0][1] if choices else None
        return gr.Dropdown(choices=choices, value=current_val), current_val, "Please enter a valid notebook name."

    new_id = notebook_mgr.create_notebook(name)
    choices = get_notebook_choices()
    return (
        gr.Dropdown(choices=choices, value=new_id),
        new_id,
        f"Created notebook: '{name}'"
    )

def handle_rename_notebook(active_id, new_name):
    """Renames the active notebook."""
    if not active_id:
        choices = get_notebook_choices()
        return gr.Dropdown(choices=choices, value=None), "No active notebook selected."
    if not new_name or not new_name.strip():
        choices = get_notebook_choices()
        return gr.Dropdown(choices=choices, value=active_id), "Please enter a valid new name."

    notebook_mgr.rename_notebook(active_id, new_name)
    choices = get_notebook_choices()
    return gr.Dropdown(choices=choices, value=active_id), f"Renamed notebook to '{new_name}'"

def handle_delete_notebook(active_id):
    """Deletes active notebook and cleans vector collection."""
    if not active_id:
        choices = get_notebook_choices()
        return gr.Dropdown(choices=choices, value=None), None, "No notebook selected to delete."

    vector_store.delete_notebook_collection(active_id)
    
    # Safely delete notebook metadata
    if active_id in notebook_mgr.metadata["notebooks"]:
        del notebook_mgr.metadata["notebooks"][active_id]
        notebook_mgr._save_metadata()

    choices = get_notebook_choices()
    next_id = choices[0][1] if choices else None
    return (
        gr.Dropdown(choices=choices, value=next_id),
        next_id,
        "Notebook and associated sources deleted."
    )

def load_notebook_data(active_id):
    """Loads active notebook sources and chat history when switched."""
    if not active_id:
        return [], "No active notebook selected."

    nb = notebook_mgr.get_notebook(active_id)
    if not nb:
        return [], "No sources uploaded yet."

    sources_text = ""
    for s in nb.get("sources", []):
        sources_text += f"• **{s['name']}** ({s['type'].upper()}) - {s['chunk_count']} chunks\n"
    if not sources_text:
        sources_text = "No sources uploaded yet."

    chat_history = []
    for msg in nb.get("chat_history", []):
        if msg["role"] == "user":
            chat_history.append((msg["content"], None))
        elif msg["role"] == "assistant" and chat_history:
            user_msg = chat_history[-1][0]
            bot_msg = msg["content"]
            if msg.get("citations"):
                bot_msg += "\n\n**Sources:** " + ", ".join(msg["citations"])
            chat_history[-1] = (user_msg, bot_msg)
    return chat_history, sources_text

def handle_file_upload(file_obj, active_id):
    """Processes uploaded file (PDF, PPTX, TXT) and adds to Chroma vector store."""
    if not active_id:
        return "Please select or create a notebook first.", ""
    if not file_obj:
        return "Please upload a valid file.", ""

    file_path = file_obj.name
    file_name = os.path.basename(file_path)
    ext = file_name.split(".")[-1].lower()

    if ext not in ["pdf", "pptx", "txt"]:
        return f"Unsupported file type: .{ext}", ""

    try:
        chunks = ingestor.process_source(
            source_input=file_path,
            source_type=ext,
            notebook_id=active_id,
            source_name=file_name
        )
        vector_store.add_chunks(active_id, chunks)
        notebook_mgr.add_source(active_id, file_name, ext, len(chunks))

        _, sources_text = load_notebook_data(active_id)
        return f"Ingested '{file_name}' ({len(chunks)} chunks)", sources_text
    except Exception as e:
        return f"Error processing file: {str(e)}", ""

def handle_url_ingest(url_input, active_id):
    """Scrapes URL text content and adds to Chroma vector store."""
    if not active_id:
        return "Please select or create a notebook first.", ""
    if not url_input or not url_input.strip():
        return "Please enter a valid URL.", ""

    try:
        url = url_input.strip()
        source_name = url.replace("https://", "").replace("http://", "").split("/")[0]
        chunks = ingestor.process_source(
            source_input=url,
            source_type="url",
            notebook_id=active_id,
            source_name=f"URL: {source_name}"
        )
        vector_store.add_chunks(active_id, chunks)
        notebook_mgr.add_source(active_id, f"URL: {source_name}", "url", len(chunks))

        _, sources_text = load_notebook_data(active_id)
        return f"Ingested URL context ({len(chunks)} chunks)", sources_text
    except Exception as e:
        return f"Error parsing URL: {str(e)}", ""

def handle_rag_chat(user_message, history, active_id, retrieval_method):
    if history is None:
        history = []

    if not user_message or not user_message.strip():
        return "", history, "", ""

    result = rag_engine.answer_question(
        notebook_id=active_id,
        query=user_message,
        retrieval_method=retrieval_method
    )

    if isinstance(result, dict):
        answer_text = result.get("answer", "No response generated.")
        sources_text = result.get("sources_formatted", result.get("sources", "No sources cited."))
        metrics_text = result.get("metrics_formatted", result.get("metrics", ""))
    else:
        answer_text = str(result)
        sources_text = ""
        metrics_text = ""

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer_text})

    return "", history, str(sources_text), str(metrics_text)

def handle_generate_report(active_id):
    """Generates Report artifact and prepares downloadable file."""
    if not active_id:
        return "Select a notebook first.", None

    result = artifact_gen.generate_report(active_id)
    if result.startswith("No documents available"):
        return f"### Warning\n{result}", None

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"report_{active_id[:8]}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result)

    return result, file_path

def handle_generate_quiz(active_id):
    """Generates Quiz artifact with Answer Key and prepares downloadable file."""
    if not active_id:
        return "Select a notebook first.", None

    quiz_text = artifact_gen.generate_quiz(active_id)
    if quiz_text.startswith("No documents available"):
        return f"### Warning\n{quiz_text}", None

    # Create a temporary .md file for the Gradio File download component
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"quiz_{active_id[:8]}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(quiz_text)

    return quiz_text, file_path

# Gradio UI Layout
with gr.Blocks(title="ITCS 5010 NotebookLM Clone Project", theme=gr.themes.Soft()) as demo:
    active_notebook_id = gr.State()
    gr.Markdown("ITCS 5010 NotebookLM Clone Project")
    gr.Markdown("A full-stack RAG application for document analysis, multisource Q&A, and study artifact generation.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Notebook Management")
            notebook_dropdown = gr.Dropdown(
                label="Select Notebook",
                choices=[],
                interactive=True
            )

            with gr.Accordion("Notebook Actions", open=False):
                new_notebook_name = gr.Textbox(label="Notebook name", placeholder="e.g., ITCS 2214 Data Structures and Algorithms")
                create_btn = gr.Button("Create Notebook", variant="primary")
                rename_btn = gr.Button("Rename Active Notebook")
                delete_btn = gr.Button("Delete Active Notebook", variant="stop")

            status_output = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Active Sources")
            sources_display = gr.Markdown("No sources loaded.")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Source Ingestion"):
                    gr.Markdown("### Add Source Material to Active Notebook")
                    with gr.Row():
                        file_input = gr.File(label="Upload File (.pdf, .pptx, .txt)", file_types=[".pdf", ".pptx", ".txt"])
                        upload_btn = gr.Button("Upload File", variant="primary")
                    with gr.Row():
                        url_input = gr.Textbox(label="Ingest Web URL", placeholder="https://en.wikipedia.org/wiki/NASA")
                        url_btn = gr.Button("Fetch & Ingest URL", variant="primary")
                    ingest_status = gr.Markdown("")

                with gr.Tab("RAG Chat"):
                    chatbot = gr.Chatbot(label="Notebook Q&A Chat", height=450)

                    with gr.Row():
                        chat_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask a question about your sources...",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Accordion("RAG Engine & Evaluation Settings", open=True):
                        retrieval_method_radio = gr.Radio(
                            choices=[("Basic Vector Search", "vector"), ("Vector Search + Reranking", "rerank")],
                            value="vector",
                            label="Retrieval Strategy (For RAG Evaluation)"
                        )
                        with gr.Row():
                            eval_time_out = gr.Textbox(label="Response Latency", interactive=False)
                            eval_chunks_out = gr.Textbox(label="Retrieved Context", interactive=False)

                with gr.Tab("Artifact Generation"):
                    gr.Markdown("### Synthesize Notebook Artifacts")
                    with gr.Row():
                        gen_report_btn = gr.Button("Generate Report (.md)", variant="primary")
                        gen_quiz_btn = gr.Button("Generate Quiz with Answer Key (.md)", variant="primary")

                    artifact_markdown = gr.Markdown(label="Artifact Preview")
                    download_file = gr.File(label="Download Generated Artifact (.md)", interactive=False)

    # Event Bindings
    demo.load(initialize_app, outputs=[notebook_dropdown, active_notebook_id])

    def on_notebook_change(nb_id):
        chat_history, sources_text = load_notebook_data(nb_id)
        return nb_id, chat_history, sources_text

    notebook_dropdown.change(
        fn=on_notebook_change,
        inputs=[notebook_dropdown],
        outputs=[active_notebook_id, chatbot, sources_display]
    )

    create_btn.click(
        fn=handle_create_notebook,
        inputs=[new_notebook_name],
        outputs=[notebook_dropdown, active_notebook_id, status_output]
    )

    rename_btn.click(
        fn=handle_rename_notebook,
        inputs=[active_notebook_id, new_notebook_name],
        outputs=[notebook_dropdown, status_output]
    )

    delete_btn.click(
        fn=handle_delete_notebook,
        inputs=[active_notebook_id],
        outputs=[notebook_dropdown, active_notebook_id, status_output]
    )

    upload_btn.click(
        fn=handle_file_upload,
        inputs=[file_input, active_notebook_id],
        outputs=[ingest_status, sources_display]
    )

    url_btn.click(
        fn=handle_url_ingest,
        inputs=[url_input, active_notebook_id],
        outputs=[ingest_status, sources_display]
    )

    send_btn.click(
        fn=handle_rag_chat,
        inputs=[chat_input, chatbot, active_notebook_id, retrieval_method_radio],
        outputs=[chat_input, chatbot, eval_time_out, eval_chunks_out]
    )

    chat_input.submit(
        fn=handle_rag_chat,
        inputs=[chat_input, chatbot, active_notebook_id, retrieval_method_radio],
        outputs=[chat_input, chatbot, eval_time_out, eval_chunks_out]
    )

    gen_report_btn.click(
        fn=handle_generate_report,
        inputs=[active_notebook_id],
        outputs=[artifact_markdown, download_file]
    )

    gen_quiz_btn.click(
        fn=handle_generate_quiz,
        inputs=[active_notebook_id],
        outputs=[artifact_markdown, download_file]
    )

if __name__ == "__main__":
    demo.queue().launch(show_error=True)