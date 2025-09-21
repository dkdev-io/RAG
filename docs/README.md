# Personal RAG Assistant

A simple, powerful Retrieval-Augmented Generation (RAG) system that helps you search and query your personal document collection using natural language.

## Features

- **Two-Phase Processing**: Assess first, then process only what you want
- **Multi-Format Support**: Text files, PDFs, Word docs, Excel, images (with OCR), and more
- **Local & Private**: Everything runs on your machine except the final LLM generation
- **Simple Interface**: Clean Streamlit web interface for querying
- **Cost-Effective**: Uses free, open-source tools with Ollama for generation

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Install Tesseract for OCR (macOS)
brew install tesseract

# Start Ollama and pull a model
ollama serve
ollama pull llama2  # or your preferred model
```

### 2. Prepare Your Documents

1. Create the source directory: `mkdir -p data/source`
2. Copy all your documents into `data/source/`
3. The system will recursively scan all files in this directory

### 3. Run the Pipeline

#### Phase 1: Assessment (Fast scan)
```bash
python src/assess.py
```
This creates `assessment_report.csv` showing what files can be processed.

#### Phase 2: Processing (Full extraction and indexing)
```bash
python src/process.py
```
This extracts content, performs OCR, and builds the vector index.

#### Phase 3: Query Interface
```bash
streamlit run src/app.py
```
Open your browser to the displayed URL and start asking questions!

## Supported File Types

- **Text**: `.txt`, `.md`, `.py`, `.js`, `.html`, `.css`, `.json`, `.xml`, `.csv`
- **Documents**: `.pdf`, `.docx`, `.doc`, `.xlsx`, `.xls`, `.pptx`, `.ppt`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.gif` (with OCR)

## Project Structure

```
rag/
├── src/
│   ├── assess.py          # Phase 1: File assessment
│   ├── process.py         # Phase 2: Content extraction & indexing
│   └── app.py            # Phase 3: Query interface
├── config/
│   └── settings.py       # Configuration settings
├── data/
│   ├── source/           # Put your documents here
│   └── processed/        # Generated vector database
├── docs/
│   └── README.md         # This file
├── requirements.txt      # Python dependencies
└── assessment_report.csv # Generated assessment report
```

## Configuration

Edit `config/settings.py` to customize:

- **Chunk size and overlap** for text splitting
- **Embedding model** (default: all-MiniLM-L6-v2)
- **Ollama model** (default: llama2)
- **Similarity threshold** for retrieval
- **File type support**

## Usage Tips

### Assessment Phase
- Review the assessment report before processing
- Check file sizes and types
- Estimated processing time helps plan the full run

### Processing Phase
- Can take hours for large document collections
- Progress bars show current status
- Logs are saved to `processing.log`
- Use `--force` flag to rebuild existing index

### Querying
- Ask natural language questions
- Be specific for better results
- Check source citations to verify answers
- Try different phrasings if results aren't good

## Example Queries

- "What are the main points from my meeting notes about the Q3 project?"
- "Find information about machine learning models in my research papers"
- "What did I write about authentication in my code documentation?"
- "Summarize the key findings from my data analysis reports"

## Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Install a model if missing
ollama pull llama2

# Try a different model if one isn't working
ollama pull mistral
```

### OCR Issues
```bash
# Test Tesseract installation
tesseract --version

# On macOS, reinstall if needed
brew reinstall tesseract
```

### Memory Issues
- Process smaller batches of files
- Adjust chunk size in `config/settings.py`
- Use a smaller embedding model

### No Results Found
- Check similarity threshold in settings
- Try broader search terms
- Verify documents were processed successfully in logs

## Advanced Usage

### Custom Source Directory
```bash
python src/assess.py --source /path/to/your/documents
```

### Rebuild Index
```bash
python src/process.py --force
```

### Change Ollama Model
Edit `config/settings.py` and change `OLLAMA_MODEL` to your preferred model.

## Performance Notes

- **Assessment**: ~100 files/second
- **Processing**: ~1-2 files/second (depends on content and OCR)
- **Querying**: ~2-5 seconds per query
- **Memory**: ~1-2GB RAM for moderate collections (10k files)

## Privacy & Security

- All document processing happens locally
- Only the final generation step uses Ollama (can be local too)
- No data is sent to external services except Ollama API
- Vector database and embeddings stay on your machine

## Cost Considerations

- All software is free and open-source
- Processing is done locally (no API costs)
- Only costs are your local compute resources
- Total one-time cost should be well under $250 as specified

## Next Steps

After getting comfortable with the basic system:

1. **Tune Settings**: Adjust chunk sizes, similarity thresholds
2. **Try Different Models**: Experiment with different Ollama models
3. **Expand Sources**: Add more document types and sources
4. **Custom Queries**: Develop query templates for common use cases