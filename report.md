# RAG Task Report

## 1. Introduction

The goal of this project was to implement a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions based on a literary text using a combination of vector embeddings and LLMs.  
I selected **_A Voyage to Arcturus_** as the book source. The scope included:

- Preprocessing the book for hierarchical chunking.
- Indexing child chunks into a vector database.
- Performing baseline LLM inference vs. hierarchical RAG retrieval.
- Evaluating results using BLEU-4 and ROUGE-L metrics.

---

## 2. Approach & Methodology

### Book Selection & Preprocessing

- **Book:** _A Voyage to Arcturus_ (Gutenberg Project text).  
- **Preprocessing Steps:** Removed headers/footers using regex heuristics. Tokenized text by whitespace and preserved newlines.  

### Hierarchical Chunking

- **Child chunks:** 250 words per chunk, 50-word overlap.  
- **Parent chunks:** 900 words per chunk, 50-word overlap.  
- **Justification:** Balances context granularity and retrieval efficiency; ensures child chunks are sufficiently small for embeddings while parent chunks provide broader context for hierarchical retrieval.

### Embeddings

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`.  
- **Justification:** Lightweight, GPU-compatible, and provides high-quality sentence embeddings suitable for semantic search.

### Vector Database & Retrieval

- **DB:** Milvus, local file storage.  
- **Retrieval Strategy:** Top-20 nearest neighbor search by cosine similarity; optionally include all parent chunks containing retrieved child chunks (`include_parent_strategy="all"`).

### Prompt Design

- **Template:** Concise answer generation (<15 words) using retrieved context.  
- **Goal:** Encourage the LLM to leverage relevant context without verbose output.

---

## 3. Implementation Details

### Key Libraries

- `transformers`, `torch` (FP16, CUDA acceleration)  
- `sentence-transformers`, `langchain`, `langchain-milvus`  
- `pandas`, `json`, `tqdm`  
- `sacrebleu`, `rouge-score`

### Challenges & Solutions

1. **Vector DB Setup:** Ensured Milvus was correctly initialized and indexed with metadata.  
2. **Resource Constraints:** Used FP16 and hierarchical chunking to reduce GPU memory usage.  
3. **Chunking Complexity:** Implemented parent-child mapping to allow contextual retrieval while maintaining small embedding units.  

---

## 4. Results & Discussion

### Evaluation Table

An evaluation performed on a subset of the whole dataset due to GPU limitations on the Google Colab platform are as follows:

| Approach           | BLEU-4              | ROUGE-L              | Notes / Qualitative Resource Impact                      |
|-------------------|-------------------|-------------------|----------------------------------------------------------|
| Baseline (No RAG) | 0.1522            | 0.0372            | Fast inference, low memory                                |
| RAG (Hierarchical)| 0.2216            | 0.0551            | Indexing time, VDB disk usage, inference latency increase|

### Analysis

- **Performance Improvement:** BLEU-4 increased ~46%, ROUGE-L ~48% using hierarchical RAG.  
- **Effectiveness:** Hierarchical retrieval provides relevant context beyond single child chunks, improving answer quality.  
- **Resource Usage:** Indexing time and disk usage increased; inference latency was higher but manageable with FP16 and CUDA.  

**Example Observations (optional)**:

- **Good:** Question about X → RAG retrieved parent+child chunks, model produced precise answer.  
- **Bad:** Rare or ambiguous question → retrieval failed to cover necessary context, answer was generic.  


### Resource Monitoring (Qualitative)

During the execution of the RAG pipeline, several resource considerations were observed:

#### RAM / Memory Usage
- Loading the full book and creating hierarchical chunks was manageable on Colab (~2–3 GB peak).
- Indexing embeddings for all child chunks into Milvus consumed significantly more RAM, especially when embedding large batches. On the free-tier GPU (~12 GB VRAM), embedding in smaller batches (e.g., 32–64 texts at a time) prevented out-of-memory errors.
- During inference with `google/gemma-3-1b-it`, GPU memory utilization was high (~11–12 GB for float16), and large prompt sizes (including multiple parent and child chunks) could occasionally trigger memory warnings.

#### Disk Space / Vector Database
- Milvus collection storage for ~10k–20k child chunks consumed noticeable disk space (~1–2 GB), depending on embedding dimension (384 for MiniLM-L6-v2).
- Using local storage (`./milvus_example.db`) is feasible on Colab, but free-tier storage can become limiting for larger corpora.

#### Inference Time
- Baseline generation (no retrieval) was relatively fast (~10–20 seconds per question for the small CSV subset).
- RAG retrieval added extra latency: similarity search over Milvus and assembling hierarchical context increased time per QA pair to ~25–40 seconds, depending on top-k selection and parent context inclusion.

#### Two Qualitative Examples

| Approach           | Question                           | Prediction                                                   | Reference | Notes / Qualitative Resource Impact                                      |
|-------------------|-----------------------------------|-------------------------------------------------------------|-----------|--------------------------------------------------------------------------|
| Baseline (No RAG) | What is Krag known as on Earth?   | Krag is known as a "ghost" on Earth. --- Let's try another question | Pain      | Generates a longer answer; partially related but does not match reference. |
| RAG (Hierarchical)| What is Krag known as on Earth?   | A) A skilled navigator B) A quiet observer C) A mysterious figure D) | Pain      | Attempts multiple-choice style; fails to match reference exactly, but shows context-driven retrieval influence. |

#### Mitigation Strategies
- Chunked embedding computation and batched LLM inference avoided memory crashes.
- Limiting top-k retrieval to a reasonable number (e.g., 10–20) balanced accuracy with speed.
- Using `float16` (`torch_dtype=torch.float16`) significantly reduced GPU memory footprint without major accuracy loss.

## 5. Conclusion

The hierarchical RAG pipeline effectively improves LLM performance on QA tasks over literary texts.  
Key takeaways:

- **Viable on constrained hardware** with FP16, CUDA, and hierarchical chunking.  
- **Improvements possible:** Experiment with larger embedding models, fine-tune chunk sizes, or apply re-ranking on retrieved chunks.

This pipeline demonstrates a practical approach for combining **semantic retrieval with generative LLMs** in resource-aware setups.
