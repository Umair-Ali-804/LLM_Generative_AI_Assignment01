"""Run every experiment required by Assignment 1, in order, from one command.

    python run_all.py                      # everything (wandb online; run `wandb login` first)
    python run_all.py --wandb_mode offline # no account needed
    python run_all.py --skip_image         # Task 3 only (the graded part)
    python run_all.py --sweep              # + learning-rate sweeps for the report

Each run writes results/<run>.json and checkpoints/<run>.pth. Model selection uses
validation metrics only; test metrics are reported once per run.
"""
import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable


def run(cmd, log_error_to=None):
    print('\n$', ' '.join(cmd), flush=True)
    if log_error_to is None:
        subprocess.run(cmd, check=True)
        return
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        Path(log_error_to).write_text(proc.stderr)
        print(f'(expected failure recorded in {log_error_to})')
        print(proc.stderr.strip().splitlines()[-1])
    else:
        print('WARNING: this run was expected to fail but succeeded')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    p.add_argument('--skip_image', action='store_true')
    p.add_argument('--skip_text', action='store_true')
    p.add_argument('--sweep', action='store_true', help='extra learning-rate runs (validation-selected)')
    p.add_argument('--img_epochs', type=int, default=5)
    p.add_argument('--txt_epochs', type=int, default=5)
    a = p.parse_args()
    w = ['--wandb_mode', a.wandb_mode]
    Path('results').mkdir(exist_ok=True)

    if not a.skip_text:
        t = [PY, 'txt_classifier.py', '--data_dir', a.data_dir, '--n_epochs', str(a.txt_epochs)] + w
        # 3.1
        run(t + ['--batch_size', '1', '--max_len', '-1', '--length_histogram', '--run_name', 'length-histogram'])
        # 3.2 (SGD, lr 5 = scaffold baseline) -- the first run doubles as the SGD row of 3.3
        run(t + ['--model', 'bag', '--optimizer', 'sgd', '--batch_size', '1', '--max_len', '-1'])
        run(t + ['--model', 'bag', '--optimizer', 'sgd', '--batch_size', '1', '--max_len', '1024'])
        run(t + ['--model', 'bag', '--optimizer', 'sgd', '--batch_size', '32', '--max_len', '1024'])
        run(t + ['--model', 'bag', '--optimizer', 'sgd', '--batch_size', '32', '--max_len', '-1'],
            log_error_to='results/error_bs32_lennone.txt')
        # 3.3 Adam (lr 1e-3) on the same settings; Adam is used from here on
        run(t + ['--model', 'bag', '--optimizer', 'adam', '--batch_size', '1', '--max_len', '-1'])
        # 3.4 LSTM (Adam), same data settings as the Adam baseline above
        run(t + ['--model', 'lstm', '--optimizer', 'adam', '--batch_size', '1', '--max_len', '-1'])
        # 3.5
        run([PY, 'analyze_pairs.py', '--data_dir', a.data_dir])
        if a.sweep:
            for lr in ('0.0001', '0.01'):
                run(t + ['--model', 'bag', '--optimizer', 'adam', '--learning_rate', lr, '--batch_size', '1',
                         '--max_len', '-1', '--run_name', f'sweep-bag-adam-lr{lr}'])
            for lr in ('0.5', '50'):
                run(t + ['--model', 'bag', '--optimizer', 'sgd', '--learning_rate', lr, '--batch_size', '1',
                         '--max_len', '-1', '--run_name', f'sweep-bag-sgd-lr{lr}'])

    if not a.skip_image:
        i = [PY, 'img_classifier.py', '--data_dir', a.data_dir, '--n_epochs', str(a.img_epochs)] + w
        run(i + ['--run_name', 'neural-the-narwhal', '--output', 'checkpoints/img_neural-the-narwhal.pth'])  # 2.1-2.3, 2.5
        run(i + ['--grayscale', '--run_name', 'grayscale', '--output', 'checkpoints/img_grayscale.pth'])      # 2.4
        run(i + ['--model', 'mlp', '--run_name', 'base-model', '--output', 'checkpoints/img_base-model.pth'])  # 2.6
        run(i + ['--model', 'cnn', '--run_name', 'new-model', '--output', 'checkpoints/img_new-model.pth'])
        if a.sweep:
            for lr in ('0.0001', '0.01'):
                run(i + ['--learning_rate', lr, '--run_name', f'sweep-mlp-sgd-lr{lr}',
                         '--output', f'checkpoints/img_sweep-mlp-sgd-lr{lr}.pth'])

    print('\nAll runs finished. Now run: python make_report.py --github <url> --drive <url> --ai_link <url>')


if __name__ == '__main__':
    main()
