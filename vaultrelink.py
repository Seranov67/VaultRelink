"""Offline Markdown link recovery. Generates a report; never changes vault notes."""
import argparse
import html
import re
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'multilingual-minilm-avx2'
LINK = re.compile(r'(?<!!)\[\[([^\]\n]+)\]\]')
WORD = re.compile(r'[^\W_]+', re.UNICODE)

def key(s):
    return re.sub(r'\s+', ' ', s.replace('\\', '/').strip()).casefold().removesuffix('.md')

def terms(s):
    return WORD.findall(s.casefold())

def load_vault(folder):
    folder = folder.resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('Vault must be a directory')
    notes, warnings = [], []
    for p in sorted(folder.rglob('*.md')):
        if p.is_symlink() or not p.resolve().is_relative_to(folder):
            warnings.append(f'Skipped symlink: {p.relative_to(folder)}')
            continue
        try:
            content = p.read_text(encoding='utf-8-sig')
        except (UnicodeError, OSError) as exc:
            warnings.append(f'Could not read {p.relative_to(folder)}: {exc}')
            continue
        rel = p.relative_to(folder).as_posix()
        notes.append({'path': rel, 'stem': p.stem, 'text': content})
    return notes, warnings

def find_broken(notes):
    paths = {key(n['path']) for n in notes}
    stems = Counter(key(n['stem']) for n in notes)
    broken = []
    for note in notes:
        for match in LINK.finditer(note['text']):
            target = match.group(1).split('|', 1)[0].split('#', 1)[0].strip()
            if not target:
                continue
            norm = key(target)
            if norm in paths or ('/' not in norm and stems[norm] == 1):
                continue
            if '/' not in norm and stems[norm] > 1:
                continue  # ambiguous name: do not present as missing
            start, end = max(0, match.start()-250), min(len(note['text']), match.end()+250)
            broken.append({'source': note['path'], 'target': target, 'raw': match.group(0),
                           'line': note['text'].count('\n', 0, match.start())+1,
                           'context': note['text'][start:end]})
    return broken

class Encoder:
    def __init__(self):
        self.tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=0, pad_token='<pad>')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(MODEL / 'onnx/model_quint8_avx2.onnx'), sess_options=opts, providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts):
        vectors = []
        for offset in range(0, len(texts), 8):
            enc = self.tokenizer.encode_batch(texts[offset:offset+8])
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {'input_ids': np.array([e.ids for e in enc], dtype=np.int64), 'attention_mask': mask}
            if 'token_type_ids' in self.inputs:
                feed['token_type_ids'] = np.array([e.type_ids for e in enc], dtype=np.int64)
            out = self.session.run(None, feed)[0]
            w = mask[..., None].astype(np.float32)
            v = (out*w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)
            vectors.extend(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9))
        return np.asarray(vectors)

def chunks(text, tokenizer):
    # Character offsets avoid silently dropping the end of long documents.
    tokenizer.no_truncation()
    try:
        raw = tokenizer.encode(text, add_special_tokens=False)
    finally:
        tokenizer.enable_truncation(max_length=128)
    offsets = raw.offsets
    if not offsets:
        return [text[:300]]
    return [text[offsets[i][0]:offsets[min(i+79, len(offsets)-1)][1]] for i in range(0, len(offsets), 65)]

def rank(notes, broken, encoder):
    # Encode each chunk with its filename; report the strongest actual excerpt per file.
    parts, owners = [], []
    for i, note in enumerate(notes):
        for chunk in chunks(note['text'], encoder.tokenizer):
            parts.append((note['stem'][:30] + '\n' + chunk, chunk))
            owners.append(i)
    if not parts or not broken:
        return []
    vectors = encoder.encode([x[0] for x in parts])
    queries = encoder.encode([b['target'] + '\n' + b['context'] for b in broken])
    results = []
    for item, q in zip(broken, queries):
        scores = vectors @ q
        candidates = []
        query_terms = set(terms(item['target'] + ' ' + item['context']))
        for i, note in enumerate(notes):
            if note['path'] == item['source']:
                continue
            indexes = [j for j, owner in enumerate(owners) if owner == i]
            if not indexes:
                continue
            best = max(indexes, key=lambda j: scores[j])
            lexical = len(query_terms & set(terms(note['stem'] + ' ' + parts[best][1]))) / max(len(query_terms), 1)
            semantic = float(scores[best])
            candidates.append({'path': note['path'], 'excerpt': parts[best][1], 'cosine': semantic,
                               'score': .85*semantic + .15*lexical})
        candidates.sort(key=lambda c: (-c['score'], c['path']))
        results.append((item, candidates[:3]))
    return results

def report(results, warnings, note_count):
    esc = html.escape
    out = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>VaultRelink report</title>',
           '<style>body{font:16px system-ui;max-width:1000px;margin:3rem auto;padding:0 1rem;color:#172139}h1{color:#4b3985}section{border:1px solid #ddd;border-radius:12px;padding:1rem;margin:1rem 0}pre{white-space:pre-wrap;background:#f4f3f8;padding:1rem}small{color:#555}</style>',
           '<h1>VaultRelink</h1><p>Recover forgotten links with local AI. Suggestions need your judgment; this report never changes notes.</p>',
           f'<p>Notes scanned: {note_count} · Broken links: {len(results)}</p>']
    for warning in warnings:
        out.append(f'<p>{esc(warning)}</p>')
    for item, candidates in results:
        out.append(f'<section><h2>{esc(item["raw"])}</h2><p>{esc(item["source"])} · line {item["line"]}</p><pre>{esc(item["context"])}</pre>')
        if not candidates:
            out.append('<p>No other note is available.</p>')
        for candidate in candidates:
            original = item['raw'][2:-2]
            target_and_heading, sep, alias = original.partition('|')
            _, heading_sep, heading = target_and_heading.partition('#')
            replacement = '[[' + candidate['path'].removesuffix('.md') + (heading_sep + heading if heading_sep else '') + (sep + alias if sep else '') + ']]'
            out.append(f'<h3>{esc(candidate["path"])}</h3><small>Combined score {candidate["score"]:.3f} · cosine {candidate["cosine"]:.3f} · ranking, not probability</small><pre>{esc(candidate["excerpt"])}</pre><p>Manual replacement: <code>{esc(item["raw"])}</code> → <code>{esc(replacement)}</code></p>')
        out.append('</section>')
    return ''.join(out) + '</html>'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vault', type=Path, help='Directory of Markdown notes (read only)')
    parser.add_argument('--output', type=Path, required=True, help='HTML report outside the vault')
    args = parser.parse_args()
    vault = args.vault.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(vault):
        parser.error('Output must be outside the vault')
    if output.exists():
        parser.error('Output already exists; choose a new path')
    notes, warnings = load_vault(vault)
    broken = find_broken(notes)
    if not notes:
        parser.error('No readable Markdown notes found')
    if broken:
        if not (MODEL / 'onnx/model_quint8_avx2.onnx').is_file():
            parser.error('Model not found. See README for setup.')
        results = rank(notes, broken, Encoder())
    else:
        results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        f.write(report(results, warnings, len(notes)))
    print(f'Notes: {len(notes)}; broken links: {len(results)}; report: {output}')

if __name__ == '__main__':
    main()
