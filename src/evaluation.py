import json, re, ast
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

def parse_question_string(s):  # duplicate-safe but kept here for independence
    s = re.sub(r'array\(\s*(.*?)\s*(?:,\s*dtype=object)?\)', r'\1', s, flags=re.DOTALL)
    s = re.sub(r'\s+', ' ', s)
    return ast.literal_eval(s)

def parse_answers_string(s):
    s_fixed = re.sub(r'}\s*{', '}, {', s)
    s_fixed = re.sub(r'array\(\s*(.*?)\s*(?:,\s*dtype=object)?\)', r'\1', s_fixed, flags=re.DOTALL)
    s_fixed = re.sub(r'\s+', ' ', s_fixed)
    return ast.literal_eval(s_fixed)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def compute_metrics(jsonl_path):
    samples = load_jsonl(jsonl_path)
    predictions = [s["prediction"] for s in samples]
    references = [s["reference"] for s in samples]

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return bleu, rouge
