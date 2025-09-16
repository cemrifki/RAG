# RAG

RAG pipeline with hierarchical chunking, Milvus vector DB, and CUDA-accelerated embedding & inference. Includes preprocessing of Project Gutenberg text, embedding/indexing with `sentence-transformers`, retrieval-augmented generation using Gemma, and evaluation with BLEU-4 & ROUGE-L metrics.


## Overview

This repo contains scripts to:

- **Preprocess** long texts into hierarchical chunks (children & parents).  
- **Index** child chunks with embeddings into **Milvus**.  
- **Run baseline QA** using Gemma without retrieval.  
- **Run RAG QA** by retrieving chunks & parents from Milvus.  
- **Evaluate** predictions against gold QA pairs using BLEU and ROUGE-L.  

CUDA (PyTorch + sentence-transformers) is supported automatically if available.


## Setup Instructions

1. Clone the repository.
2. Create Python virtual environment:

```bash
   python -m venv my_venv
```

Activate it:

```bash
source my_venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

I tested this with Python 3.12 and CUDA-enabled PyTorch 2.4.0.

You will also need a HuggingFace token. After obtaining one, run the following on Linux:

```bash
echo 'export HF_TOKEN="YOUR_HF_TOKEN_HERE"' >> ~/.bashrc
source ~/.bashrc
```
## Project Structure

project/
├── main.py                 # Entrypoint
├── requirements.txt
└── src/
    ├── preprocessing.py    # Gutenberg cleanup + hierarchical chunking
    ├── indexing.py         # Embed & store chunks in Milvus
    ├── rag_pipeline.py     # Baseline & RAG pipelines
    ├── evaluation.py       # BLEU & ROUGE metrics
    └── utils.py            # Helpers: parsing, context assembly, prompt template

## Usage
Run the main pipeline end-to-end (preprocessing → indexing → QA → evaluation):

```bash
python main.py
```
If you want to run modules separately, you can import them.

Example: run only preprocessing:

```bash
python -m src.preprocessing
```

## Outputs

Preprocessed chunks: data/chunks/parents.jsonl & children.jsonl

Predictions: outputs/predictions_baseline.jsonl & outputs/predictions_rag.jsonl

Evaluation: printed BLEU & ROUGE-L scores

## Notes

- All the code was written by Cem Rifki Aydin
- CUDA is automatically used if available (torch.cuda.is_available()).
- Uses Gemma 3.1 IT model for QA (google/gemma-3-1b-it).
- Retrieval powered by Milvus (langchain-milvus).
