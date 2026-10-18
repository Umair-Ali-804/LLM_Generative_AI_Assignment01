# Assignment 1 - PyTorch Foundations: Image and Text Classification

Generative AI & LLM, Fall 2026. Python scripts only (no notebooks).
Task 3 (text classification) is the graded part; Task 2 (images) is fully implemented too.

## Files

| File | Purpose |
|---|---|
| `txt_classifier.py` | Task 3: histogram (3.1), truncate/pad transforms (3.2), SGD/Adam (3.3), LSTM (3.4) |
| `img_classifier.py` | Task 2: wandb logging, validation, grayscale, captioned predictions, CNN model |
| `analyze_pairs.py` | Task 3.5: prints real/fake pairs and measures style differences |
| `run_all.py` | Runs every required experiment in order |
| `make_report.py` | Builds `Report.pdf` from the measured results |
| `data/` | Put the course CSVs and the `parrot/ narwhal/ axolotl/` image folders here |
| `results/`, `checkpoints/` | Written by the runs (JSON metrics, plots, `.pth` checkpoints) |

## Setup (Python 3.10-3.12)

    python -m pip install -r requirements.txt
    wandb login                      # or use --wandb_mode offline / disabled

For a GPU, install the matching torch 2.5.1 / torchvision 0.20.1 build from https://pytorch.org first.

## Run everything

    python run_all.py                       # all tasks, wandb online
    python run_all.py --skip_image          # Task 3 only
    python run_all.py --wandb_mode offline  # no wandb account needed
    python run_all.py --sweep               # adds learning-rate sweeps for the report
    python make_report.py --github <url> --drive <url> --ai_link <url> --wandb_links "<urls>"

`--sweep` runs are compared on validation accuracy only; test accuracy is reported once per run
and never used to choose settings.

## Individual commands (Task 3)

    python txt_classifier.py --batch_size 1 --max_len -1 --length_histogram     # 3.1
    python txt_classifier.py --optimizer sgd --batch_size 1  --max_len -1       # 3.2.a
    python txt_classifier.py --optimizer sgd --batch_size 1  --max_len 1024
    python txt_classifier.py --optimizer sgd --batch_size 32 --max_len 1024
    python txt_classifier.py --optimizer sgd --batch_size 32 --max_len -1       # 3.2.b (expected error)
    python txt_classifier.py --optimizer adam --batch_size 1 --max_len -1       # 3.3 (Adam, lr 1e-3)
    python txt_classifier.py --model lstm --batch_size 1 --max_len -1           # 3.4
    python analyze_pairs.py                                                      # 3.5

Defaults: `--model bag`, `--optimizer adam` (lr 1e-3; with `--optimizer sgd` the lr defaults to the
scaffold's 5.0), 5 epochs, seed 42. Other options: `--embed_dim --hidden_dim --num_layers
--unidirectional --dropout --clip_norm --learning_rate --device --wandb_mode --run_name --output`.
`PadSequence` is unchanged from the scaffold, so batch size 32 with `--max_len -1` fails with a
`torch.stack` size-mismatch error, as Task 3.2.b requires.

LSTM model: `nn.Embedding` (padding_idx = pad) -> bidirectional `nn.LSTM(batch_first=True)` on packed
sequences -> masked `AdaptiveMaxPool1d` over real timesteps -> dropout -> linear.

## Individual commands (Task 2)

    python img_classifier.py --run_name neural-the-narwhal                 # 2.1-2.3, 2.5 (MLP, color)
    python img_classifier.py --grayscale --run_name grayscale              # 2.4
    python img_classifier.py --model mlp --run_name base-model             # 2.6
    python img_classifier.py --model cnn --run_name new-model              # 2.6 (graph from the handout)

Other options: `--n_epochs --batch_size --learning_rate --optimizer {sgd,adam} --data_dir --output --seed
--wandb_mode {online,offline,disabled}`. On the last epoch the first batch of train/val/test is logged to
wandb with `<predicted> / <true>` captions.

## Model checkpoints

Google Drive link: PASTE_LINK_HERE   (upload the files in `checkpoints/`)

## Git workflow

Commit incrementally (baseline run, wandb logging, padding transforms, optimizer switch, LSTM, report).
`data/`, `wandb/` and `*.pth` are git-ignored.
