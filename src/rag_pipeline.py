import json, re, ast, time
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_milvus import Milvus
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import parse_question_string, parse_answers_string, assemble_context, prompt_template

max_new_tokens = 17

def run_baseline_csv(hf_model, qa_csv_path, out_dir):
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model, torch_dtype=torch.float16, device_map="auto"
    )
    df = pd.read_csv(qa_csv_path, sep=",", engine='python', dtype=str)
    df['question'] = df['question'].apply(parse_question_string)
    df['answers'] = df['answers'].apply(parse_answers_string)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions_baseline.jsonl"
    open(preds_path, "w").close()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Baseline QA"):
        qn, answers_list = row['question'], row['answers']
        question, reference = qn['text'], (answers_list[0]['text'] if answers_list else None)
        prompt = f"Answer the question as best as you can:\n\nQuestion: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).replace("\n", " ")
        gen = re.sub(r"\s+", " ", gen).strip()

        res = {"question": question, "prediction": gen, "reference": reference}
        with open(preds_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("Wrote baseline predictions to", preds_path)

def run_rag_milvus_csv(uri, embedding_model, hf_model, chunks_path, parents_path, qa_csv_path, out_dir, top_k, include_parent_strategy, collection_name):
    chunks_path = Path(chunks_path)
    children = [json.loads(line) for line in open(chunks_path / "children.jsonl", encoding="utf-8")]

    embedder = SentenceTransformerEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    vector_store = Milvus(
        embedding_function=embedder,
        collection_name=collection_name,
        connection_args={"uri": uri},
        index_params={"index_type": "FLAT", "metric_type": "COSINE"}
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float16, device_map="auto")

    parents = {}
    if include_parent_strategy in ("top1", "all"):
        with open(parents_path, "r", encoding="utf-8") as pf:
            for pl in pf:
                p = json.loads(pl)
                parents[p["parent_id"]] = p["text"]

    df = pd.read_csv(qa_csv_path, sep=",", engine='python', dtype=str)
    df['question'] = df['question'].apply(parse_question_string)
    df['answers'] = df['answers'].apply(parse_answers_string)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions_rag.jsonl"
    open(preds_path, "w").close()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="RAG QA"):
        question, reference = row['question']['text'], (row['answers'][0]['text'] if row['answers'] else None)
        q_emb = embedder.embed_query(question)
        hits = vector_store.similarity_search_by_vector(q_emb, k=top_k, include_metadata=True)

        child_hits, parent_ids = [], set()
        for h in hits:
            meta = h.metadata
            child_hits.append({"chunk_text": meta.get("chunk_text", ""), "parent_ids": meta.get("parent_ids", [])})
            parent_ids.update(meta.get("parent_ids", []))

        parent_texts = []
        if include_parent_strategy == "top1" and child_hits:
            for pid in child_hits[0]["parent_ids"]:
                if pid in parents: parent_texts.append(parents[pid])
        elif include_parent_strategy == "all":
            for pid in parent_ids:
                if pid in parents: parent_texts.append(parents[pid])

        context = assemble_context(child_hits, include_parent_texts=parent_texts)
        prompt = prompt_template(context, question)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).replace("\n", " ")
        gen = re.sub(r"\s+", " ", gen).strip()

        res = {"question": question, "prediction": gen, "reference": reference}
        with open(preds_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("Wrote RAG predictions to", preds_path)
