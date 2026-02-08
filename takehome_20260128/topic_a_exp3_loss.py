"""
Experiment 3: Loss Function for Distillation

Tests how different loss functions affect subliminal learning quality.

Conditions:
1. KL divergence (default — softmax + KL div)
2. MSE on raw logits
3. MSE on softmax probabilities
4. Cosine similarity loss
"""
import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms
import os


# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
N_MODELS = 25
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE = 1024
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))


# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list[int]):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int]):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle
            else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        self._mkperm() if self.shuffle else None
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────── train / distill ────────────────────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(student, teacher, idx, src_x, epochs: int, loss_type: str = "kl"):
    """Distill with configurable loss function."""
    opt = t.optim.Adam(student.parameters(), lr=LR)
    F = nn.functional
    for _ in tqdm.trange(epochs, desc=f"distill ({loss_type})"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]

            if loss_type == "kl":
                loss = F.kl_div(
                    F.log_softmax(out, -1),
                    F.softmax(tgt, -1),
                    reduction="batchmean",
                )
            elif loss_type == "mse_logits":
                loss = F.mse_loss(out, tgt)
            elif loss_type == "mse_probs":
                loss = F.mse_loss(F.softmax(out, -1), F.softmax(tgt, -1))
            elif loss_type == "cosine":
                loss = 1 - F.cosine_similarity(out, tgt, dim=-1).mean()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


# ───────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    t.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    rand_imgs = t.rand_like(train_x) * 2 - 1

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # Train reference and teacher once
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ref_state = reference.state_dict()
    ref_acc_mean = np.mean(accuracy(reference, test_x, test_y))

    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(ref_state)
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    # Run each loss function condition
    loss_types = ["kl", "mse_logits", "mse_probs", "cosine"]
    loss_labels = [
        "KL Divergence\n(default)",
        "MSE\n(raw logits)",
        "MSE\n(softmax probs)",
        "Cosine\nSimilarity",
    ]

    results = {}
    for loss_type, label in zip(loss_types, loss_labels):
        print(f"\n{'='*60}")
        print(f"Loss: {label.replace(chr(10), ' ')}")
        print(f"{'='*60}")

        # Reset seed for reproducibility of student init
        t.manual_seed(SEED)
        np.random.seed(SEED)

        # Fresh student from reference init
        student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student.load_state_dict(ref_state)

        # Distill on ghost logits only
        distill(student, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL,
                loss_type=loss_type)

        # Evaluate
        acc = accuracy(student, test_x, test_y)
        results[label] = acc
        print(f"  Accuracy: {np.mean(acc):.4f} ± {ci_95(acc):.4f}")

        del student
        t.cuda.empty_cache() if DEVICE == "cuda" else None

    # ─────────────────────── plot results ──────────────────────────────────
    cond_names = list(results.keys())
    means = [np.mean(results[c]) for c in cond_names]
    cis = [ci_95(results[c]) for c in cond_names]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["C0", "C1", "C2", "C3"]
    bars = ax.bar(cond_names, means, yerr=cis, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(ref_acc_mean, ls=":", c="black", alpha=0.5,
               label=f"Chance ({ref_acc_mean:.0%})")
    ax.axhline(means[0], ls="--", c="C0", alpha=0.4,
               label=f"KL baseline ({means[0]:.0%})")
    ax.set_ylabel("Test accuracy (digit classification)", fontsize=13)
    ax.set_title("Exp 3: Effect of Loss Function on Subliminal Learning", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, max(means) * 1.25)
    plt.tight_layout()

    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    outpath = f"plots_a/{script_name}_results.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {outpath}")

    # Print summary table
    df = pd.DataFrame({
        "Loss Function": [c.replace("\n", " ") for c in cond_names],
        "Mean Accuracy": [f"{m:.4f}" for m in means],
        "95% CI": [f"±{c:.4f}" for c in cis],
    })
    print("\n" + df.to_string(index=False))
