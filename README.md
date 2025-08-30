# My RAG Bot

This project is a Retrieval-Augmented Generation (RAG) bot that leverages LangChain, Chroma vector database, and HuggingFace models to answer questions based on your own documents (CSV, PDF, TXT). It loads, chunks, embeds, and indexes your data, then uses a local LLM to answer queries with context retrieved from your documents.

## Features
- Ingests CSV, PDF, and TXT files
- Chunks documents to avoid token overflow
- Embeds and stores chunks in a Chroma vector database
- Uses HuggingFace LLM (e.g., Flan-T5) for answer generation
- Retrieval-augmented QA pipeline

## Project Structure
```
main.py                # Main script to run the RAG bot
requirements.txt       # Python dependencies
chroma_db/             # Chroma vector database files
/data/
  ToTo.csv             # Example CSV file
  sg60-national-day.txt# Example TXT file
  test.pdf             # Example PDF file
```

## Setup
1. **Clone the repository**
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   python3 -m venv my-rag-bot
   source my-rag-bot/bin/activate
   pip install -r requirements.txt
   ```
3. **Add your data**
   - Place your CSV, PDF, and TXT files in the `data/` directory.
   - Update the file paths in `main.py` if needed.

## Running the Bot
Run the main script:
```bash
python main.py
```

You can modify the `query` variable in `main.py` to test different questions.

## Customization
- **Model**: Change the HuggingFace model in `load_hf_llm()` (e.g., to a larger Flan-T5 variant).
- **Chunk size/overlap**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `main.py` for your data.
- **Top-K retrieval**: Change `TOP_K` to control how many chunks are retrieved for each query.

## Requirements
- Python 3.8+
- See `requirements.txt` for all Python dependencies

## Notes
- The first run will download HuggingFace models and may take a while.
- The Chroma vector DB is persisted in `chroma_db/`.
- For large datasets or models, ensure you have sufficient RAM and disk space.

## License
This project is for educational and research purposes.

