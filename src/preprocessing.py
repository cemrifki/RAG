from pathlib import Path
import re
import json

def strip_gutenberg_header_footer(text: str) -> str:
    start_re = r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK[^\n]*\*\*\*'
    end_re = r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK[^\n]*\*\*\*'
    s = re.search(start_re, text, flags=re.IGNORECASE)
    e = re.search(end_re, text, flags=re.IGNORECASE)
    if s and e:
        return text[s.end():e.start()]
    lines = text.splitlines()
    return '\n'.join(lines[200:-200])

def tokenize_by_words(text: str):
    return text.split()

def create_hierarchical_chunks(text, child_size=300, parent_size=1000, overlap=50):
    words = tokenize_by_words(text)
    children, parents = [], []

    child_step = child_size - overlap
    for i in range(0, max(1, len(words) - child_size + 1), child_step):
        chunk_words = words[i:i + child_size]
        children.append((' '.join(chunk_words), i, i + child_size))

    parent_step = parent_size - overlap
    for i in range(0, max(1, len(words) - parent_size + 1), parent_step):
        pwords = words[i:i + parent_size]
        parents.append((' '.join(pwords), i, i + parent_size))

    child_meta = []
    for idx, (ctext, cstart, cend) in enumerate(children):
        parent_ids = [
            pid for pid, (ptext, pstart, pfinish) in enumerate(parents)
            if (cstart >= pstart and cend <= pfinish)
        ]
        child_meta.append({
            'child_id': idx, 'text': ctext,
            'start': cstart, 'end': cend, 'parent_ids': parent_ids
        })
    parent_meta = [
        {'parent_id': pid, 'text': ptext, 'start': pstart, 'end': pfinish}
        for pid, (ptext, pstart, pfinish) in enumerate(parents)
    ]
    return parent_meta, child_meta

def preprocess(book_txt, out_dir, child_size, parent_size, overlap):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = Path(book_txt).read_text(encoding='utf-8')
    clean = strip_gutenberg_header_footer(text)
    parents, children = create_hierarchical_chunks(
        clean,
        child_size=int(child_size),
        parent_size=int(parent_size),
        overlap=int(overlap)
    )
    with open(out_dir / 'parents.jsonl', 'w', encoding='utf-8') as f:
        for p in parents:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    with open(out_dir / 'children.jsonl', 'w', encoding='utf-8') as f:
        for c in children:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(f"Wrote {len(parents)} parents and {len(children)} children to {out_dir}")
