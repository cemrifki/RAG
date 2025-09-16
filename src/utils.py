import re, ast

def parse_question_string(s):
    s = re.sub(r'array\(\s*(.*?)\s*(?:,\s*dtype=object)?\)', r'\1', s, flags=re.DOTALL)
    s = re.sub(r'\s+', ' ', s)
    return ast.literal_eval(s)

def parse_answers_string(s):
    s_fixed = re.sub(r'}\s*{', '}, {', s)
    s_fixed = re.sub(r'array\(\s*(.*?)\s*(?:,\s*dtype=object)?\)', r'\1', s_fixed, flags=re.DOTALL)
    s_fixed = re.sub(r'\s+', ' ', s_fixed)
    return ast.literal_eval(s_fixed)

def assemble_context(child_hits, include_parent_texts=None, max_chars=2000):
    pieces, cur = [], 0
    if include_parent_texts:
        pieces.extend(include_parent_texts)
    for c in child_hits:
        if cur + len(c['chunk_text']) > max_chars: break
        pieces.append(c['chunk_text'])
        cur += len(c['chunk_text'])
    return "\n\n".join(pieces)

def prompt_template(context, question):
    return f"""Answer the question using the context. If unsure, make your best guess.
Keep it concise (<15 words). Shorter is better.

Context:
{context}

Question: {question}
Answer:"""
