"""Task 3.5 support: print real/fake article pairs and measure stylistic differences.

Usage: python analyze_pairs.py --data_dir data --n_pairs 3
Writes results/pair_analysis.json (used by make_report.py).
Note: the articles are unfiltered news text; read with care.
"""
import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from txt_classifier import tokenize


def load(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def style_stats(articles):
    toks = [tokenize(a) for a in articles]
    n = len(articles)
    total = sum(len(t) for t in toks)
    return {
        'mean_tokens': total / n,
        'digit_tokens_per_1000': 1000 * sum(tok.isdigit() for t in toks for tok in t) / total,
        'quote_marks_per_1000': 1000 * sum(a.count('"') + a.count('\u201c') + a.count('\u201d') for a in articles) / total,
        'frac_with_bracket_placeholder': sum(bool(re.search(r'\[[^\]]{1,30}\]', a)) for a in articles) / n,
        'frac_with_newlines_in_body': sum(a.strip().count('\n') > 1 for a in articles) / n,
        'type_token_ratio': sum(len(set(t)) / max(len(t), 1) for t in toks) / n,
    }


def top_words(real, fake, k=12, min_count=8):
    cr = Counter(w for a in real for w in set(tokenize(a)))
    cf = Counter(w for a in fake for w in set(tokenize(a)))
    nr, nf = len(real), len(fake)
    rows = []
    for w in set(cr) | set(cf):
        if cr[w] + cf[w] < min_count or not re.search(r'\w', w):
            continue
        lo = math.log((cf[w] + 1) / (nf + 2)) - math.log((cr[w] + 1) / (nr + 2))
        rows.append((lo, w, cr[w], cf[w]))
    rows.sort()
    return ([(w, a, b) for _, w, a, b in rows[:k]], [(w, a, b) for _, w, a, b in rows[::-1][:k]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--n_pairs', type=int, default=3)
    p.add_argument('--chars', type=int, default=500)
    p.add_argument('--out', default='results/pair_analysis.json')
    args = p.parse_args()

    rows = load(Path(args.data_dir) / 'txt_train.csv')
    real = [r['article'] for r in rows if r['label_idx'] == '0']
    fake = [r['article'] for r in rows if r['label_idx'] == '1']

    print('=' * 80)
    shown = 0
    for i in range(0, len(rows) - 1, 2):
        a, b = rows[i], rows[i + 1]
        if {a['label_idx'], b['label_idx']} != {'0', '1'}:
            continue
        r, f = (a, b) if a['label_idx'] == '0' else (b, a)
        print(f'PAIR {shown + 1} (train rows {i}, {i + 1})\n--- REAL ---\n{r["article"][:args.chars]}\n'
              f'--- FAKE ---\n{f["article"][:args.chars]}\n' + '=' * 80)
        shown += 1
        if shown >= args.n_pairs:
            break

    stats = {'real': style_stats(real), 'fake': style_stats(fake)}
    fake_words, real_words = top_words(real, fake)
    out = {'n_real': len(real), 'n_fake': len(fake), 'stats': stats,
           'words_more_common_in_real': real_words, 'words_more_common_in_fake': fake_words}
    print(json.dumps(out, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
