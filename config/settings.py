"""Configuration settings for the Personal RAG Assistant."""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DEFAULT_SOURCE_DIR = BASE_DIR / "data" / "source"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ASSESSMENT_REPORT = BASE_DIR / "assessment_report.csv"

# Dynamic source directory (can be changed by user)
SOURCE_DIR = DEFAULT_SOURCE_DIR

# Vector database settings
VECTOR_DB_PATH = PROCESSED_DIR / "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama settings
OLLAMA_MODEL = "llama2:latest"  # You can change this to your preferred model
OLLAMA_BASE_URL = "http://localhost:11434"

# File type mappings
SUPPORTED_EXTENSIONS = {
    # Text files
    '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv',
    # Documents
    '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt',
    # Images
    '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif',
    # Archives (we'll extract and process contents)
    '.zip', '.tar', '.gz', '.7z'
}

# OCR settings
TESSERACT_CONFIG = '--oem 3 --psm 6'

# Retrieval settings
MAX_RETRIEVED_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.25