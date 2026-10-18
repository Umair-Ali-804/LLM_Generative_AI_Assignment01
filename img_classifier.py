"""Task 2: parrot / narwhal / axolotl image classification (ungraded, but fully implemented).

Extends the supplied baseline with:
  * wandb logging (per-batch loss vs. examples seen; one epoch-level log call; test metrics at the end)
  * validation loss / accuracy every epoch
  * --grayscale  (Task 2.4)
  * last-epoch logging of the first batch of train/val/test with "<pred> / <true>" captions (Task 2.5)
  * --model cnn  (Task 2.6 computation graph) next to the original --model mlp
  * --optimizer {sgd,adam}
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

img_size = (256, 256)
num_labels = 3
class_names = ['parrot', 'narwhal', 'axolotl']  # overwritten from the CSV when available

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

RGB_MEAN, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
GRAY_MEAN, GRAY_STD = [0.449], [0.226]  # luminance-weighted ImageNet statistics


class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_root = Path(csv_file).resolve().parent
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        parts = Path(img_name).parts
        relative = Path(*parts[parts.index("data") + 1:]) if "data" in parts else Path(img_name)
        with Image.open(self.data_root / relative) as im:
            image = im.convert("RGB")
        label = int(self.data_frame.loc[idx, "label_idx"])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data(batch_size, data_dir, grayscale):
    steps = [T.Grayscale(num_output_channels=1)] if grayscale else []
    mean, std = (GRAY_MEAN, GRAY_STD) if grayscale else (RGB_MEAN, RGB_STD)
    steps += [
        T.ToTensor(),
        T.Resize(min(img_size), antialias=True),  # smallest side -> 256
        T.CenterCrop(img_size),                   # centre crop 256x256
        T.Normalize(mean=mean, std=std),
    ]
    transform_img = T.Compose(steps)
    data_dir = Path(data_dir)
    train_data = CsvImageDataset(data_dir / 'img_train.csv', transform_img)
    val_data = CsvImageDataset(data_dir / 'img_val.csv', transform_img)
    test_data = CsvImageDataset(data_dir / 'img_test.csv', transform_img)

    global class_names
    df = train_data.data_frame
    if 'label' in df.columns:
        class_names = [df.loc[df.label_idx == i, 'label'].iloc[0] for i in sorted(df.label_idx.unique())]

    loaders = (DataLoader(train_data, batch_size=batch_size, shuffle=True),
               DataLoader(val_data, batch_size=batch_size),
               DataLoader(test_data, batch_size=batch_size))
    for X, y in loaders[0]:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    return loaders, (mean, std)


# -------------------------------------------------------------------------- models
class NeuralNetwork(nn.Module):
    """Original baseline: flatten -> 512 -> 512 -> 3 with ReLU."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size[0] * img_size[1] * in_channels, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, num_labels))

    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) tensor."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNet(nn.Module):
    """Task 2.6 graph:
    (B,C,256,256) -Conv k=4,s=4-> (B,128,64,64) -LN- -Conv k=7,p=3-> -LN-
    -Conv k=1 (128->256)- GELU -Conv k=1 (256->128)- -AvgPool k=2-> (B,128,32,32)
    -Flatten-> (B,131072) -Linear-> (B,3)
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=4),
            LayerNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            LayerNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, num_labels))

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------- logging
class Logger:
    """Thin wandb wrapper; a no-op when --wandb_mode disabled."""
    def __init__(self, mode, project, name, config):
        self.run = None
        if mode != 'disabled':
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, name=name, config=config, mode=mode)
            wandb.define_metric('examples_seen')
            wandb.define_metric('batch_loss', step_metric='examples_seen')
            wandb.define_metric('epoch')
            for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc'):
                wandb.define_metric(k, step_metric='epoch')

    def log(self, data):
        if self.run:
            self.run.log(data)

    def finish(self):
        if self.run:
            self.run.finish()


def log_first_batch(loader, split, model, mean, std, logger):
    """Task 2.5: caption '<predicted> / <true>' for the first batch of a split."""
    if not logger.run:
        return
    X, y = next(iter(loader))
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device)).argmax(1).cpu()
    denorm = T.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
    imgs = []
    for i in range(len(y)):
        xi = torch.clamp(denorm(X[i]), min=0, max=1)
        imgs.append(logger.wandb.Image(xi, caption=f"{class_names[pred[i]]} / {class_names[y[i]]}",
                                       mode="L" if xi.shape[0] == 1 else "RGB"))
    logger.log({f"{split}_first_batch": imgs})


# ------------------------------------------------------------------ train / evaluate
def train_one_epoch(dataloader, model, loss_fn, optimizer, logger, examples_seen, batch_losses):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)          # CrossEntropyLoss -> mean over the batch
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        examples_seen += len(y)
        batch_losses.append((examples_seen, loss.item()))   # also kept locally for make_report.py
        logger.log({"batch_loss": loss.item(), "examples_seen": examples_seen})
        if batch % 10 == 0:
            print(f"Train loss = {loss.item():>7f}  [{examples_seen:>5d}]")
    return examples_seen


def evaluate(dataloader, dataname, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    avg_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item() * len(y)       # sample-weighted over the dataset
            correct += (pred.argmax(1) == y).float().sum().item()
    avg_loss /= size
    acc = correct / size
    print(f"{dataname} accuracy = {100 * acc:>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")
    return avg_loss, acc


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    print(f"Using {device} device")
    in_channels = 1 if args.grayscale else 3
    run_name = args.run_name or f"{args.model}-{'gray' if args.grayscale else 'rgb'}-{args.optimizer}"
    config = dict(vars(args), run_name=run_name, device=device)
    logger = Logger(args.wandb_mode, args.wandb_project, run_name, config)

    (train_dl, val_dl, test_dl), (mean, std) = get_data(args.batch_size, args.data_dir, args.grayscale)
    model = (NeuralNetwork if args.model == 'mlp' else ConvNet)(in_channels).to(device)
    n_params = count_parameters(model)
    print(model, f"\nTrainable parameters: {n_params:,}")
    if logger.run:
        logger.run.config.update({"num_parameters": n_params}, allow_val_change=True)

    loss_fn = nn.CrossEntropyLoss()
    opt_cls = torch.optim.SGD if args.optimizer == 'sgd' else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=args.learning_rate)

    history, batch_losses, examples_seen, start = [], [], 0, time.perf_counter()
    for t in range(args.n_epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        examples_seen = train_one_epoch(train_dl, model, loss_fn, optimizer, logger, examples_seen, batch_losses)
        train_loss, train_acc = evaluate(train_dl, "Train", model, loss_fn)
        val_loss, val_acc = evaluate(val_dl, "Validation", model, loss_fn)
        rec = dict(epoch=t + 1, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_acc=val_acc)
        history.append(rec)
        logger.log(rec)                                   # single epoch-level call
        if t == args.n_epochs - 1:                        # last epoch only
            for split, dl in (("train", train_dl), ("val", val_dl), ("test", test_dl)):
                log_first_batch(dl, split, model, mean, std, logger)

    test_loss, test_acc = evaluate(test_dl, "Test", model, loss_fn)
    logger.log({"test_loss": test_loss, "test_acc": test_acc})
    print("Done!")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Saved PyTorch Model State to {out}")

    result = dict(run_name=run_name, model=args.model, grayscale=args.grayscale,
                  optimizer=args.optimizer, learning_rate=args.learning_rate,
                  batch_size=args.batch_size, n_epochs=args.n_epochs, seed=args.seed,
                  num_parameters=n_params, runtime_sec=time.perf_counter() - start,
                  final_train_acc=history[-1]['train_acc'], final_val_acc=history[-1]['val_acc'],
                  test_loss=test_loss, test_acc=test_acc, history=history, batch_losses=batch_losses,
                  checkpoint=str(out))
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"img_{run_name}.json").write_text(json.dumps(result, indent=2))
    if logger.run:
        logger.run.summary.update({k: v for k, v in result.items() if k not in ('history', 'batch_losses')})
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--n_epochs', default=5, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--data_dir', '--data-dir', default='data')
    parser.add_argument('--output', default='checkpoints/img_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--grayscale', action='store_true', help='Task 2.4: single-channel input')
    parser.add_argument('--run_name', default=None,
                        help='e.g. neural-the-narwhal (2.1), base-model / new-model (2.6.b)')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--wandb_project', default='img_classifier')
    main(parser.parse_args())
