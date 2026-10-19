# Assignment 1 - PyTorch Foundations: Image and Text Classification

**Course:** Generative AI & LLM, Fall 2026
**Student:** Umair Ali (MSDS24066)
**Implementation:** Python `.py` scripts only (no notebooks).

Task 3 (text classification) is the graded part. Task 2 (image classification) is fully implemented as well.

## Links

| What | Link |
|---|---|
| GitHub (private) | https://github.com/Umair-Ali-804/LLM_Generative_AI_Assignment01 |
| Model checkpoints (Google Drive) | https://drive.google.com/drive/folders/1PsVnznjIOs5PLRfHBxuf5OWr_HffjEzt |
| W&B - image runs | https://forge.coreweave.com/wandb/aliumair64488-information-technology-university/img_classifier |
| W&B - text runs | https://forge.coreweave.com/wandb/aliumair64488-information-technology-university/txt_classifier |

The report (`Report.pdf`) is in the repository root.

## Files

| File | Purpose |
|---|---|
| `txt_classifier.py` | Task 3: length histogram (3.1), truncate/pad transforms (3.2), SGD vs Adam (3.3), LSTM model (3.4) |
| `img_classifier.py` | Task 2: wandb logging, validation, grayscale, captioned predictions, CNN model |
| `analyze_pairs.py` | Task 3.5: prints real/fake article pairs and measures style differences |
| `run_all.py` | Runs every required experiment in order with one command |
| `requirements.txt` | Python dependencies |
| `data/` | Put the course CSVs and the `parrot/ narwhal/ axolotl/` image folders here (see `data/README.txt`) |
| `results/`, `checkpoints/` | Created by the runs: JSON metrics, plots and `.pth` checkpoints |

## Data

Copy the course data into `data/` so that this layout exists:

    data/txt_train.csv  data/txt_val.csv  data/txt_test.csv
    data/img_train.csv  data/img_val.csv  data/img_test.csv
    data/parrot/  data/narwhal/  data/axolotl/

Do not re-split the CSVs. Every script takes `--data_dir` if the data is somewhere else.

## Setup

### Local machine (Python 3.10-3.12)

    python -m pip install -r requirements.txt
    wandb login                       # or use --wandb_mode offline / disabled

For a GPU, install the matching torch 2.5.1 / torchvision 0.20.1 build from https://pytorch.org first.

### Google Colab (GPU runtime)

Colab already ships with PyTorch, and the pinned `torch==2.5.1` in `requirements.txt` may not install on
Colab's newer Python. Do **not** run `pip install -r requirements.txt` there. Install only what is missing:

    from google.colab import drive
    drive.mount('/content/drive')

    %cd /content/drive/MyDrive/Assignment1_Solution
    !pip install -q wandb
    !wandb login
    !python run_all.py

Set the runtime to GPU via Runtime -> Change runtime type -> T4 GPU. The notebook is only a launcher
for Colab and is not part of the submission.

## Run everything

    python run_all.py                       # all tasks, wandb online
    python run_all.py --skip_image          # Task 3 only (the graded part)
    python run_all.py --skip_text           # Task 2 only
    python run_all.py --wandb_mode offline  # no wandb account needed
    python run_all.py --sweep               # adds learning-rate sweeps
    python run_all.py --data_dir /path/to/data --img_epochs 5 --txt_epochs 5

`--sweep` runs are compared on validation accuracy only. Test accuracy is reported once per run and is
never used to choose settings. Every run writes `results/<run>.json` and `checkpoints/<run>.pth`.

## Task 3 commands (text classification)

    python txt_classifier.py --batch_size 1 --max_len -1 --length_histogram     # 3.1 histogram
    python txt_classifier.py --optimizer sgd --batch_size 1  --max_len -1       # 3.2.a runtime
    python txt_classifier.py --optimizer sgd --batch_size 1  --max_len 1024
    python txt_classifier.py --optimizer sgd --batch_size 32 --max_len 1024
    python txt_classifier.py --optimizer sgd --batch_size 32 --max_len -1       # 3.2.b (expected error)
    python txt_classifier.py --optimizer adam --batch_size 1 --max_len -1       # 3.3 (Adam, lr 1e-3)
    python txt_classifier.py --model lstm --batch_size 1 --max_len -1           # 3.4 LSTM
    python analyze_pairs.py --n_pairs 3                                          # 3.5 pair analysis

Add `--wandb_mode online` to any `txt_classifier.py` command to log to the `txt_classifier` W&B project
(the default for this script is `disabled`; `run_all.py` turns it on).

Defaults: `--model bag`, `--optimizer adam` (lr 1e-3; with `--optimizer sgd` the lr defaults to the
scaffold's 5.0), `--batch_size 1`, `--max_len -1`, 5 epochs, seed 42.
Other options: `--embed_dim --hidden_dim --num_layers --unidirectional --dropout --clip_norm
--learning_rate --device --run_name --output --results_dir --wandb_project`.

`PadSequence` is unchanged from the scaffold, so batch size 32 with `--max_len -1` fails with a
`torch.stack` size-mismatch error, as Task 3.2.b requires. That error is expected.

**LSTM model (3.4):** `nn.Embedding` (padding_idx = pad) -> bidirectional `nn.LSTM(batch_first=True)` on
packed sequences -> masked `AdaptiveMaxPool1d` over real timesteps -> dropout -> linear layer.

## Task 2 commands (image classification)

    python img_classifier.py --run_name neural-the-narwhal                 # 2.1-2.3, 2.5 (MLP, color)
    python img_classifier.py --grayscale --run_name grayscale              # 2.4
    python img_classifier.py --model mlp --run_name base-model             # 2.6 original model
    python img_classifier.py --model cnn --run_name new-model              # 2.6 new model (CNN graph)

Defaults: SGD, learning rate 1e-3, batch size 8, 5 epochs, seed 42, W&B mode `online`.
Other options: `--n_epochs --batch_size --learning_rate --optimizer {sgd,adam} --data_dir --output --seed
--results_dir --wandb_mode {online,offline,disabled} --wandb_project`.
On the last epoch the first batch of train/val/test is logged to W&B with `<predicted> / <true>` captions.

## Outputs

- `results/*.json` - per-run hyperparameters, per-epoch metrics, final train/val/test accuracy, runtime
- `results/article_length_histogram.png` - Task 3.1 histogram
- `checkpoints/*.pth` - trained models (uploaded to the Google Drive folder above)

## Git workflow

The history is committed incrementally (baseline run, wandb logging, padding transforms, optimizer
switch, LSTM, image CNN, report). `data/`, `wandb/` and `*.pth` are git-ignored.
