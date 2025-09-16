#!/usr/bin/env python3
"""Entrypoint for running preprocessing, indexing, RAG, and evaluation."""

from src.preprocessing import preprocess
from src.indexing import index_embeddings
from src.rag_pipeline import run_baseline_csv, run_rag_milvus_csv
from src.evaluation import compute_metrics


import os
from huggingface_hub import login


if __name__ == "__main__":
    """
    Main entry point for running the full RAG pipeline:
    1. Preprocess the book text into hierarchical chunks.
    2. Index the chunks into a Milvus vector database.
    3. Run a baseline QA model without retrieval.
    4. Run a RAG pipeline with retrieval from Milvus.
    5. Evaluate and compare the results using BLEU and ROUGE-L metrics.
    """

    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    # Step 1: Preprocess text
    preprocess(
        book_txt="data/A_voyage_to_Arcturus.txt",
        out_dir="data/chunks",
        child_size=250,
        parent_size=900,
        overlap=50
    )

    # Step 2: Index embeddings into Milvus
    index_embeddings(
        chunks_dir="data/chunks",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        uri="./milvus_example.db",
        collection_name="children"
    )

    # Step 3: Run baseline (no RAG)
    run_baseline_csv(
        hf_model="google/gemma-3-1b-it",
        qa_csv_path="data/Arcturus_QA_Pairs.csv",
        out_dir="outputs/"
    )

    # Step 4: Run RAG pipeline
    run_rag_milvus_csv(
        uri="./milvus_example.db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        hf_model="google/gemma-3-1b-it",
        chunks_path="data/chunks",
        parents_path="data/chunks/parents.jsonl",
        qa_csv_path="data/Arcturus_QA_Pairs.csv",
        out_dir="outputs/",
        top_k=20,
        include_parent_strategy="all",
        collection_name="children"
    )

    # Step 5: Evaluate results
    baseline_preds_jsonl = "outputs/predictions_baseline.jsonl"
    rag_preds_jsonl = "outputs/predictions_rag.jsonl"

    bleu_base, rouge_base = compute_metrics(baseline_preds_jsonl)
    bleu_rag, rouge_rag = compute_metrics(rag_preds_jsonl)

    print("\n=== Evaluation Results ===")
    print(f"Baseline -> BLEU: {bleu_base:.4f}, ROUGE-L: {rouge_base:.4f}")
    print(f"RAG      -> BLEU: {bleu_rag:.4f}, ROUGE-L: {rouge_rag:.4f}")
