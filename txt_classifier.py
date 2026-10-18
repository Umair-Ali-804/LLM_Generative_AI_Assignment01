"""Task 3: real vs. generated news classification.

Extends the supplied scaffold:
  3.1  --length_histogram       raw token-length histogram (train split)
  3.2  TruncateToMaxLen / PadToMaxLen transforms
  3.3  --optimizer {sgd,adam}   (Adam, lr 1e-3, is the default from here on)
  3.4  --model {bag,lstm}       LSTM classifier using nn.Embedding + max pooling
"""
import argparse
import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- data
def tokenize(text):
    """Lowercase word/punctuation tokenizer; no downloads required."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class Vocabulary:
    def __init__(self, texts):
        counts = Counter(token for text in texts for token in tokenize(text))
        self.stoi = {'<UNK>': 0, '<PAD>': 1}
        for token in sorted(counts, key=lambda t: (-counts[t], t)):
            if token not in self.stoi:
                self.stoi[token] = len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, 0)

    def __len__(self):
        return len(self.stoi)


class CsvTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        with open(csv_file, encoding='utf-8', newline='') as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row['article']  # only the article text is a model input
        if self.transform is not None:
            text = self.transform(text)
        return text, int(row['label_idx'])


class CorpusInfo:
    def __init__(self, dataset):
        self.tokenizer = tokenize
        self.vocab = Vocabulary(row['article'] for row in dataset.rows)  # train text only
        self.oov_token, self.pad_token = '<UNK>', '<PAD>'
        self.oov_idx, self.pad_idx = 0, 1
        self.vocab_size = len(self.vocab)
        self.num_labels = len({row['label_idx'] for row in dataset.rows})


class TextTransform:
    def __init__(self, tokenizer, vocab):
        self.tokenizer, self.vocab = tokenizer, vocab

    def __call__(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]


class TruncateToMaxLen:
    """Keep at most the first `max_len` tokens of an article."""
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tokens):
        return tokens[:self.max_len]


class PadToMaxLen:
    """Right-pad an article with `pad_idx` until it has exactly `max_len` tokens."""
    def __init__(self, max_len, pad_idx):
        self.max_len, self.pad_idx = max_len, pad_idx

    def __call__(self, tokens):
        return tokens + [self.pad_idx] * (self.max_len - len(tokens))


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class PadSequence:
    """Stack sequences as supplied by the transforms (no automatic padding).

    Intentionally unchanged from the scaffold: with variable-length articles and
    batch_size > 1 this raises the torch.stack error needed for Task 3.2.b.
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        text = torch.stack([torch.tensor(x, dtype=torch.long) for x, _ in batch])
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
        return text, labels


# -------------------------------------------------------------------------- models
class TextClassificationModel(nn.Module):
    """Baseline: mean of token embeddings -> linear."""
    def __init__(self, vocab_size, embed_dim, num_class, pad_idx):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean', padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_class)
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()
        nn.init.uniform_(self.fc.weight, -0.5, 0.5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, text):
        return self.fc(self.embedding(text))


class LSTMClassifier(nn.Module):
    """nn.Embedding -> (bi)LSTM over real tokens -> masked max-pool over time -> linear.

    Padding is handled twice: padding_idx keeps the pad embedding at zero, and
    pack_padded_sequence makes the LSTM skip padded steps entirely, so the final
    states are never contaminated by pad tokens. Max pooling (AdaptiveMaxPool1d)
    runs over the real timesteps only.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, pad_idx,
                 num_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_class)

    def forward(self, text):                                   # text: (B, T)
        mask = text != self.pad_idx                            # (B, T) True on real tokens
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.embedding(text), lengths,
                                      batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=text.size(1))  # (B, T, H')
        out = out.masked_fill(~mask.unsqueeze(-1), -1e9)       # padded steps never win the max
        pooled = self.pool(out.transpose(1, 2)).squeeze(-1)    # (B, H')
        return self.fc(self.dropout(pooled))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------- helpers
class Logger:
    """Thin wandb wrapper; a no-op unless --wandb_mode is online/offline."""
    def __init__(self, mode, project, name, config):
        self.run = None
        if mode != 'disabled':
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, name=name, config=config, mode=mode)

    def log(self, data):
        if self.run:
            self.run.log(data)

    def image(self, path):
        return self.wandb.Image(path)

    def summary(self, data):
        if self.run:
            self.run.summary.update(data)

    def finish(self):
        if self.run:
            self.run.finish()


def evaluate(loader, model, criterion, device):
    model.eval()
    correct = count = 0
    loss_sum = 0.0
    with torch.no_grad():
        for text, labels in loader:
            text, labels = text.to(device), labels.to(device)
            logits = model(text)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return loss_sum / count, correct / count


def length_histogram(raw_train, tokenizer, out_path, logger):
    """Task 3.1: raw (pre-truncation, pre-padding) article lengths, training split, in tokens."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lengths = np.array([len(tokenizer(r['article'])) for r in raw_train.rows])
    labels = np.array([int(r['label_idx']) for r in raw_train.rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, lengths.max(), 41)
    ax.hist([lengths[labels == 0], lengths[labels == 1]], bins=bins, stacked=True,
            label=['real (0)', 'fake (1)'], color=['#3b6ea5', '#d98b3a'], edgecolor='white')
    ax.axvline(1024, color='k', ls='--', lw=1, label='max_len = 1024')
    ax.set_xlabel('Article length (tokens, lowercase word/punctuation tokenizer)')
    ax.set_ylabel('Number of articles')
    ax.set_title(f'Raw article lengths, training split (n={len(lengths)})')
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    stats = {'split': 'train', 'unit': 'tokens', 'n_articles': int(len(lengths)),
             'min': int(lengths.min()), 'median': float(np.median(lengths)),
             'mean': float(lengths.mean()), 'p95': float(np.percentile(lengths, 95)),
             'max': int(lengths.max()),
             'frac_over_1024': float((lengths > 1024).mean()),
             'mean_len_real': float(lengths[labels == 0].mean()),
             'mean_len_fake': float(lengths[labels == 1].mean())}
    print(json.dumps(stats, indent=2))
    stats_path = Path(out_path).with_suffix('.json')
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.log({'article_length_histogram': logger.image(str(out_path)) if logger.run else None})
    logger.summary(stats)
    print(f'Saved {out_path} and {stats_path}')


def build_optimizer(name, params, lr):
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description='Task 3 text classifier')
    parser.add_argument('--data_dir', '--data-dir', default='data')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=1)
    parser.add_argument('--max_len', '--max-len', type=int, default=-1)
    parser.add_argument('--n_epochs', '--epochs', type=int, default=5)
    parser.add_argument('--model', choices=['bag', 'lstm'], default='bag',
                        help='bag = original EmbeddingBag baseline, lstm = Task 3.4 model')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam',
                        help='adam is the default from Task 3.3 onward; use sgd for Tasks 3.2/3.3')
    parser.add_argument('--learning_rate', '--learning-rate', type=float, default=None,
                        help='default: 5.0 for sgd (scaffold value), 1e-3 for adam')
    parser.add_argument('--clip_norm', type=float, default=0.1, help='gradient-norm clip (scaffold value)')
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--unidirectional', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--length_histogram', '--length-histogram', action='store_true')
    parser.add_argument('--hist_out', default='results/article_length_histogram.png')
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--output', default=None, help='checkpoint path (default checkpoints/<run_name>.pth)')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--wandb_mode', choices=['disabled', 'online', 'offline'], default='disabled')
    parser.add_argument('--wandb_project', default='txt_classifier')
    args = parser.parse_args()
    if args.batch_size < 1 or args.n_epochs < 1 or args.max_len == 0 or args.max_len < -1:
        parser.error('Use positive batch size/epochs and max_len=-1 or a positive integer.')

    lr = args.learning_rate if args.learning_rate is not None else (5.0 if args.optimizer == 'sgd' else 1e-3)
    run_name = args.run_name or (f'{args.model}-{args.optimizer}-bs{args.batch_size}-'
                                 f'len{"none" if args.max_len == -1 else args.max_len}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(vars(args), learning_rate=lr, run_name=run_name, device=device)
    logger = Logger(args.wandb_mode, args.wandb_project, run_name, config)

    data_dir = Path(args.data_dir)
    raw_train = CsvTextDataset(data_dir / 'txt_train.csv')
    corpus_info = CorpusInfo(raw_train)

    if args.length_histogram:
        length_histogram(raw_train, corpus_info.tokenizer, args.hist_out, logger)
        logger.finish()
        return

    transforms = [TextTransform(corpus_info.tokenizer, corpus_info.vocab)]
    if args.max_len != -1:
        transforms += [TruncateToMaxLen(args.max_len), PadToMaxLen(args.max_len, corpus_info.pad_idx)]
    transform = Compose(transforms)
    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = CsvTextDataset(data_dir / f'txt_{split}.csv', transform)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'),
                                    collate_fn=PadSequence(corpus_info.pad_idx))

    if args.model == 'bag':
        model = TextClassificationModel(corpus_info.vocab_size, args.embed_dim,
                                        corpus_info.num_labels, corpus_info.pad_idx)
    else:
        model = LSTMClassifier(corpus_info.vocab_size, args.embed_dim, args.hidden_dim,
                               corpus_info.num_labels, corpus_info.pad_idx, args.num_layers,
                               not args.unidirectional, args.dropout)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f'{run_name}: {n_params:,} trainable parameters, device={device}, '
          f'optimizer={args.optimizer}, lr={lr}')
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), lr)

    history = []
    start = time.perf_counter()
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        for text, labels in loaders['train']:
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(text), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
        train_loss, train_acc = evaluate(loaders['train'], model, criterion, device)
        val_loss, val_acc = evaluate(loaders['val'], model, criterion, device)
        rec = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_acc=val_acc)
        history.append(rec)
        logger.log(rec)
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
    test_loss, test_acc = evaluate(loaders['test'], model, criterion, device)
    if device == 'cuda':
        torch.cuda.synchronize()
    runtime = time.perf_counter() - start
    print(f'Test loss={test_loss:.4f} test accuracy={test_acc:.4f}')
    print(f'Training and evaluation runtime: {runtime:.3f} seconds')

    ckpt = Path(args.output or f'checkpoints/{run_name}.pth')
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'vocab': corpus_info.vocab.stoi,
                'pad_idx': corpus_info.pad_idx, 'args': config}, ckpt)

    result = dict(run_name=run_name, model=args.model, optimizer=args.optimizer, learning_rate=lr,
                  batch_size=args.batch_size, max_len=args.max_len, n_epochs=args.n_epochs,
                  seed=args.seed, device=device, num_parameters=n_params, runtime_sec=runtime,
                  final_train_acc=history[-1]['train_acc'], final_val_acc=history[-1]['val_acc'],
                  test_loss=test_loss, test_acc=test_acc, history=history, checkpoint=str(ckpt))
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f'{run_name}.json').write_text(json.dumps(result, indent=2))
    logger.log({'test_loss': test_loss, 'test_acc': test_acc, 'runtime_sec': runtime})
    logger.summary({k: v for k, v in result.items() if k != 'history'})
    logger.finish()


if __name__ == '__main__':
    main()
