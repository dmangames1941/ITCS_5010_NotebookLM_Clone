import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

METADATA_FILE = "data/metadata.json"
DATA_DIR = "data"

class NotebookManager:
    def __init__(self, metadata_path: str = METADATA_FILE):
        self.metadata_path = metadata_path
        self._ensure_storage_exists()
        self.metadata = self._load_metadata()

    def _ensure_storage_exists(self) -> None:
        """Ensures the base data directory and metadata file exist."""
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump({"notebooks": {}}, f, indent=2)

    def _load_metadata(self) -> Dict:
        """Loads metadata from disk."""
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"notebooks": {}}

    def _save_metadata(self) -> None:
        """Saves active metadata to disk."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def create_notebook(self, name: str) -> str:
        """Create a new notebook and sets up its data directory."""
        notebook_id = f"nb_{int(datetime.now().timestamp() * 1000)}"
        notebook_dir = os.path.join(DATA_DIR, notebook_id)
        os.makedirs(notebook_dir, exist_ok=True)

        notebook_data = {
            "id": notebook_id,
            "name": name.strip() if name.strip() else "Untitled Notebook",
            "create_at": datetime.now().isoformat(),
            "sources": [],
            "chat_history": [],
            "artifacts": {}
        }

        self.metadata["notebooks"][notebook_id] = notebook_data
        self._save_metadata()
        return notebook_id

    def list_notebooks(self) ->List[Dict[str, str]]:
        """Returns a list of all notebooks with ID and name for UI dropdowns."""
        return [
            {"id": nb_id, "name": nb["name"]}
            for nb_id, nb in self.metadata["notebooks"].items()
        ]

    def get_notebook(self, notebook_id: str) -> Optional[Dict]:
        """Retrieves a specific notebook's metadata."""
        return self.metadata["notebooks"].get(notebook_id)

    def rename_notebook(self, notebook_id: str, new_name: str) -> bool:
        """Renames an existing notebook."""
        if notebook_id in self.metadata["notebooks"]:
            self.metadata["notebooks"][notebook_id]["name"] = new_name.strip()
            self._save_metadata()
            return True
        return False

    def add_source(self, notebook_id: str, source_name: str, source_type: str, chunk_count: int) -> bool:
        """Registers an ingested source file or URL under a notebook."""
        notebook = self.get_notebook(notebook_id)
        if notebook:
            source_info = {
                "source_id": f"src_{int(datetime.now().timestamp() * 1000)}",
                "name": source_name,
                "type": source_type,
                "chunk_count": chunk_count,
                "added_at": datetime.now().isoformat()
            }
            notebook["sources"].append(source_info)
            self._save_metadata()
            return True
        return False

    def add_chat_message(self, notebook_id: str, role: str, content: str, citations: Optional[List[str]] = None) -> bool:
        """Appends a message to the notebook's persistent chat history."""
        notebook = self.get_notebook(notebook_id)
        if notebook:
            msg = {
                "role": role,
                "content": content,
                "citations": citations or [],
                "timestamp": datetime.now().isoformat()
            }
            notebook["chat_history"].append(msg)
            self._save_metadata()
            return True
        return False

    def get_chat_history(self, notebook_id: str) -> List[Dict]:
        notebook = self.get_notebook(notebook_id)
        return notebook.get("chat_history", []) if notebook else []

    def set_artifact_path(self, notebook_id: str, artifact_type: str, file_path: str) -> bool:
        """Registers a generated artifact file path (e.g., 'report' or 'quiz')."""
        notebook = self.get_notebook(notebook_id)
        if notebook:
            notebook["artifacts"][artifact_type] = file_path
            self._save_metadata()
            return True
        return False