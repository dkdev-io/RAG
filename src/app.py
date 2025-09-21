#!/usr/bin/env python3
"""
Streamlit Web Interface for Personal RAG Assistant
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL,
    MAX_RETRIEVED_CHUNKS, SIMILARITY_THRESHOLD, SOURCE_DIR, PROCESSED_DIR
)
from src.folder_manager import FolderManager, render_folder_selector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """The main RAG system combining retrieval and generation."""

    def __init__(self, folder_manager: FolderManager):
        self.folder_manager = folder_manager
        self.vector_db_path = folder_manager.get_vector_db_path()
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.load_vector_database()

    @st.cache_resource
    def load_vector_database(_self):
        """Load the vector database from disk."""
        if not _self.vector_db_path.exists():
            st.error(f"Vector database not found at {_self.vector_db_path}")
            st.error("Please run the processing pipeline first:")
            st.code("Use the 'Manage Files' tab to process your selected folder")
            st.stop()

        try:
            # Load embedding model
            _self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            # Load FAISS index
            _self.index = faiss.read_index(str(_self.vector_db_path / "index.faiss"))

            # Load chunks metadata
            with open(_self.vector_db_path / "chunks.json", 'r', encoding='utf-8') as f:
                _self.chunks = json.load(f)

            logger.info(f"Loaded vector database with {len(_self.chunks)} chunks")
            return True

        except Exception as e:
            st.error(f"Error loading vector database: {e}")
            st.stop()

    def retrieve_relevant_chunks(self, query: str, max_chunks: int = MAX_RETRIEVED_CHUNKS) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a query."""
        if not self.embedding_model or not self.index:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, max_chunks)

        # Filter by similarity threshold and prepare results
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= SIMILARITY_THRESHOLD and idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                relevant_chunks.append(chunk)

        return relevant_chunks

    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> Optional[str]:
        """Generate an answer using Ollama."""
        if not relevant_chunks:
            return "No relevant information found in your documents."

        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            source_file = Path(chunk['source_file']).name
            context_parts.append(f"[Source {i+1}: {source_file}]\n{chunk['text']}\n")

        context = "\n".join(context_parts)

        # Prepare the prompt
        prompt = f"""Based on the following information from the user's personal documents, please provide a comprehensive answer to their question. Include specific details and cite the sources when relevant.

Context from documents:
{context}

Question: {query}

Please provide a detailed answer based on the information above. If the information is insufficient to fully answer the question, please say so and suggest what additional information might be needed."""

        # Call Ollama API
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error generating response: API returned status {response.status_code}"

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error connecting to Ollama: {e}"

    def query(self, question: str) -> Dict[str, Any]:
        """Main query method that combines retrieval and generation."""
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question)

        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)

        # Prepare sources
        sources = []
        for chunk in relevant_chunks:
            source_file = Path(chunk['source_file']).name
            sources.append({
                'file': source_file,
                'similarity': round(chunk['similarity_score'], 3),
                'chunk_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            })

        return {
            'answer': answer,
            'sources': sources,
            'num_chunks_retrieved': len(relevant_chunks)
        }


def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return True, model_names
        else:
            return False, []
    except requests.exceptions.RequestException:
        return False, []


def main():
    st.set_page_config(
        page_title="Personal RAG Assistant",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Personal RAG Assistant")
    st.markdown("Ask questions about your personal document collection")

    # Initialize folder manager
    folder_manager = FolderManager()

    # Add navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Query Documents", "Select Folder", "Manage Files", "System Status"])

    with tab4:  # System Status tab
        show_system_status(folder_manager)

    with tab3:  # File Management tab
        show_file_management(folder_manager)

    with tab2:  # Folder Selection tab
        show_folder_selection(folder_manager)

    with tab1:  # Query tab (main functionality)
        show_query_interface(folder_manager)


def show_folder_selection(folder_manager: FolderManager):
    """Display folder selection interface."""
    st.header("📁 Select Source Folder")
    render_folder_selector()


def show_system_status(folder_manager: FolderManager):
    """Display system status and connection information."""
    st.header("System Status")

    # Check Ollama connection
    with st.spinner("Checking Ollama connection..."):
        ollama_connected, available_models = check_ollama_connection()

    if not ollama_connected:
        st.error("❌ Cannot connect to Ollama")
        st.markdown("""
        **Please ensure Ollama is running:**
        1. Install Ollama from https://ollama.ai
        2. Start Ollama: `ollama serve`
        3. Pull a model: `ollama pull llama2` (or your preferred model)
        4. Refresh this page
        """)
        return False

    st.success(f"✅ Connected to Ollama")

    # Show available models
    if available_models:
        if OLLAMA_MODEL not in available_models:
            st.warning(f"Model '{OLLAMA_MODEL}' not found. Available models: {', '.join(available_models)}")
            st.markdown(f"Install the model with: `ollama pull {OLLAMA_MODEL}`")
        else:
            st.info(f"Using model: {OLLAMA_MODEL}")

    # Check vector database status
    vector_db_path = folder_manager.get_vector_db_path()
    if vector_db_path.exists():
        st.success("✅ Vector database found")
        try:
            with open(vector_db_path / "chunks.json", 'r') as f:
                chunks = json.load(f)
            st.info(f"📄 Total chunks: {len(chunks)}")
        except:
            st.warning("Vector database found but could not read chunk count")
    else:
        st.error("❌ Vector database not found")
        st.info("Run assessment and processing first")

    # System information
    st.markdown("### Configuration")
    st.info(f"🤖 Model: {OLLAMA_MODEL}")
    st.info(f"🔍 Max retrieved chunks: {MAX_RETRIEVED_CHUNKS}")
    st.info(f"📊 Similarity threshold: {SIMILARITY_THRESHOLD}")
    st.info(f"📁 Source directory: {folder_manager.get_source_directory()}")
    st.info(f"💾 Vector database: {vector_db_path}")

    return True


def show_file_management(folder_manager: FolderManager):
    """Display file management interface for bulk processing."""
    st.header("File Management")

    source_dir = folder_manager.get_source_directory()
    assessment_report = folder_manager.get_assessment_report_path()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Files")

        # Show source directory contents
        if source_dir.exists():
            files = list(source_dir.rglob('*'))
            files = [f for f in files if f.is_file()]

            if files:
                st.info(f"Found {len(files)} files in source directory")

                # Show file types breakdown
                extensions = {}
                for file in files:
                    ext = file.suffix.lower()
                    extensions[ext] = extensions.get(ext, 0) + 1

                if extensions:
                    st.markdown("**File types:**")
                    for ext, count in sorted(extensions.items()):
                        st.text(f"{ext or 'no extension'}: {count} files")
            else:
                st.warning("No files found in source directory")
        else:
            st.error(f"Source directory not found: {source_dir}")

    with col2:
        st.subheader("Processing Status")

        # Check assessment report
        if assessment_report.exists():
            try:
                assessment_df = pd.read_csv(assessment_report)
                st.success(f"✅ Assessment report found ({len(assessment_df)} files)")

                # Show status breakdown
                status_counts = assessment_df['status'].value_counts()
                for status, count in status_counts.items():
                    if status == 'Readable':
                        st.success(f"✅ {status}: {count} files")
                    elif status == 'Incompatible':
                        st.warning(f"⚠️ {status}: {count} files (may be convertible)")
                    else:
                        st.error(f"❌ {status}: {count} files")

            except Exception as e:
                st.error(f"Error reading assessment report: {e}")
        else:
            st.warning("No assessment report found")

    # Processing actions
    st.markdown("---")
    st.subheader("Processing Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Assessment", type="secondary"):
            with st.spinner("Running file assessment..."):
                try:
                    env = os.environ.copy()
                    env["RAG_SOURCE_DIR"] = str(source_dir)
                    env["RAG_ASSESSMENT_REPORT"] = str(assessment_report)

                    result = subprocess.run([
                        sys.executable, str(Path(__file__).parent / "assess.py")
                    ], capture_output=True, text=True, timeout=60, env=env)

                    if result.returncode == 0:
                        st.success("Assessment completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Assessment failed: {result.stderr}")

                except subprocess.TimeoutExpired:
                    st.error("Assessment timed out")
                except Exception as e:
                    st.error(f"Error running assessment: {e}")

    with col2:
        if st.button("Process Files", type="secondary"):
            with st.spinner("Processing files and building vector database..."):
                try:
                    env = os.environ.copy()
                    env["RAG_SOURCE_DIR"] = str(source_dir)
                    env["RAG_ASSESSMENT_REPORT"] = str(assessment_report)
                    env["RAG_VECTOR_DB_PATH"] = str(folder_manager.get_vector_db_path())

                    result = subprocess.run([
                        sys.executable, str(Path(__file__).parent / "process.py"), "--force"
                    ], capture_output=True, text=True, timeout=300, env=env)

                    if result.returncode == 0:
                        st.success("Processing completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Processing failed: {result.stderr}")

                except subprocess.TimeoutExpired:
                    st.error("Processing timed out")
                except Exception as e:
                    st.error(f"Error running processing: {e}")

    with col3:
        if st.button("Clear Database", type="secondary"):
            if st.checkbox("Confirm database deletion"):
                try:
                    vector_db_path = folder_manager.get_vector_db_path()
                    if vector_db_path.exists():
                        import shutil
                        shutil.rmtree(vector_db_path)
                        st.success("Vector database cleared!")
                        st.rerun()
                    else:
                        st.info("No database to clear")
                except Exception as e:
                    st.error(f"Error clearing database: {e}")

    # Instructions
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. **Select folder** in the 'Select Folder' tab: `{}`
    2. **Run Assessment** to analyze file compatibility
    3. **Process Files** to extract content and build searchable database
    4. **Query Documents** in the main tab to search your files

    **Supported file types:** Text, PDF, Word docs, Excel, images (with OCR), and more.
    **New:** Incompatible files are automatically converted when possible!
    """.format(source_dir))


def show_query_interface(folder_manager: FolderManager):
    """Display the main query interface."""
    st.header("Query Your Documents")

    # Show current folder
    st.info(f"📁 Searching in: {folder_manager.get_source_directory()}")

    # Check if system is ready
    if not show_system_status(folder_manager):
        return

    # Initialize RAG system
    try:
        with st.spinner("Loading vector database..."):
            rag_system = RAGSystem(folder_manager)
        st.success(f"✅ Loaded {len(rag_system.chunks)} document chunks")
    except Exception as e:
        st.error(f"Failed to load vector database: {e}")
        st.info("Please run assessment and processing first in the 'Manage Files' tab")
        return

    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What are the main topics covered in my notes about machine learning?"
    )

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and generating answer..."):
                result = rag_system.query(query.strip())

            # Display answer
            st.markdown("### Answer")
            st.markdown(result['answer'])

            # Display sources
            if result['sources']:
                st.markdown("### Sources")
                st.info(f"Found {result['num_chunks_retrieved']} relevant chunks")

                for i, source in enumerate(result['sources'], 1):
                    with st.expander(f"Source {i}: {source['file']} (similarity: {source['similarity']})"):
                        st.text(source['chunk_preview'])
            else:
                st.warning("No relevant sources found.")


if __name__ == "__main__":
    main()