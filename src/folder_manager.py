#!/usr/bin/env python3
"""
Folder Management for RAG System
Handles dynamic source directory selection and configuration.
"""

import json
import os
from pathlib import Path
from typing import Optional, List
import streamlit as st

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import BASE_DIR, DEFAULT_SOURCE_DIR, PROCESSED_DIR


class FolderManager:
    """Manages source folder selection and configuration."""

    def __init__(self):
        self.config_file = BASE_DIR / "folder_config.json"
        self.current_source_dir = self.load_source_directory()

    def load_source_directory(self) -> Path:
        """Load the currently selected source directory from config."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    source_path = Path(config.get('source_directory', DEFAULT_SOURCE_DIR))
                    if source_path.exists():
                        return source_path
            except Exception:
                pass
        return DEFAULT_SOURCE_DIR

    def save_source_directory(self, source_dir: Path) -> bool:
        """Save the selected source directory to config."""
        try:
            config = {'source_directory': str(source_dir)}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.current_source_dir = source_dir
            return True
        except Exception:
            return False

    def get_source_directory(self) -> Path:
        """Get the current source directory."""
        return self.current_source_dir

    def set_source_directory(self, source_dir: str) -> bool:
        """Set a new source directory."""
        try:
            path = Path(source_dir).expanduser().resolve()
            if path.exists() and path.is_dir():
                return self.save_source_directory(path)
            return False
        except Exception:
            return False

    def get_recent_folders(self) -> List[str]:
        """Get list of recently used folders."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    recent = config.get('recent_folders', [])
                    # Filter out non-existent directories
                    return [folder for folder in recent if Path(folder).exists()]
            except Exception:
                pass
        return []

    def add_to_recent_folders(self, folder_path: str) -> None:
        """Add a folder to the recent folders list."""
        try:
            recent_folders = self.get_recent_folders()
            folder_path = str(Path(folder_path).resolve())

            # Remove if already in list
            if folder_path in recent_folders:
                recent_folders.remove(folder_path)

            # Add to beginning
            recent_folders.insert(0, folder_path)

            # Keep only last 10
            recent_folders = recent_folders[:10]

            # Update config
            config = {'source_directory': str(self.current_source_dir), 'recent_folders': recent_folders}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass

    def get_suggested_folders(self) -> List[str]:
        """Get suggested folder locations."""
        suggestions = []
        home = Path.home()

        # Common document folders
        common_folders = [
            "Documents",
            "Desktop",
            "Downloads",
            "Dropbox",
            "Google Drive",
            "OneDrive",
            "iCloud Drive",
            "Library/CloudStorage"
        ]

        for folder in common_folders:
            path = home / folder
            if path.exists() and path.is_dir():
                suggestions.append(str(path))

        # Add recent folders
        suggestions.extend(self.get_recent_folders())

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for item in suggestions:
            if item not in seen:
                seen.add(item)
                unique_suggestions.append(item)

        return unique_suggestions

    def get_vector_db_path(self) -> Path:
        """Get the vector database path for current source directory."""
        # Create unique vector DB for each source directory
        source_name = self.current_source_dir.name
        source_hash = abs(hash(str(self.current_source_dir))) % 10000
        return PROCESSED_DIR / f"vector_db_{source_name}_{source_hash}"

    def get_assessment_report_path(self) -> Path:
        """Get the assessment report path for current source directory."""
        source_name = self.current_source_dir.name
        source_hash = abs(hash(str(self.current_source_dir))) % 10000
        return BASE_DIR / f"assessment_report_{source_name}_{source_hash}.csv"


def render_folder_selector() -> Optional[str]:
    """Render the folder selection interface in Streamlit."""
    st.subheader("📁 Select Source Folder")

    folder_manager = FolderManager()
    current_folder = str(folder_manager.get_source_directory())

    # Show current folder
    st.info(f"Current folder: {current_folder}")

    # Simple folder browser interface
    if 'current_browse_path' not in st.session_state:
        st.session_state.current_browse_path = str(Path.home())

    # Navigation
    current_path = Path(st.session_state.current_browse_path)

    # Up button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⬆️ Up") and current_path.parent != current_path:
            st.session_state.current_browse_path = str(current_path.parent)
            st.rerun()

    with col2:
        st.text(f"📂 {current_path}")

    # List directories
    try:
        directories = [d for d in current_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        directories.sort(key=lambda x: x.name.lower())

        # Display directories as buttons
        for directory in directories[:20]:  # Limit to 20 for performance
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📁 {directory.name}", key=f"dir_{hash(str(directory))}"):
                    st.session_state.current_browse_path = str(directory)
                    st.rerun()
            with col2:
                if st.button("Select", key=f"sel_{hash(str(directory))}"):
                    if folder_manager.set_source_directory(str(directory)):
                        folder_manager.add_to_recent_folders(str(directory))
                        st.success(f"✅ Selected: {directory}")
                        st.rerun()

    except PermissionError:
        st.error("Permission denied to access this directory")
    except Exception as e:
        st.error(f"Error reading directory: {e}")

    return str(folder_manager.get_source_directory())