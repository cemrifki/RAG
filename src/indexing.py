from pathlib import Path
import json
import torch
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_milvus import Milvus

def load_children(chunks_dir):
    children = []
    with open(Path(chunks_dir) / 'children.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            children.append(json.loads(line))
    return children

def index_embeddings(chunks_dir, embedding_model, uri="./milvus_example.db", collection_name="children"):
    children = load_children(chunks_dir)
    texts = [c['text'] for c in children]

    embedding_function = SentenceTransformerEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    vector_store = Milvus(
        embedding_function=embedding_function,
        collection_name=collection_name,
        connection_args={"uri": uri},
        index_params={"index_type": "FLAT", "metric_type": "COSINE"}
    )

    metadatas = [{
        "child_id": str(c["child_id"]),
        "parent_ids": json.dumps(c.get("parent_ids", [])),
        "chunk_text": c["text"],
    } for c in children]

    vector_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"Indexed {len(children)} chunks into Milvus collection '{collection_name}'.")
