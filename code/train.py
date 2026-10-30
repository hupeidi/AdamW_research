"""Small, offline-reproducible AdamW course experiment (Python >=3.9)."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CONFIGS = {
    "adam": ("adam", 0.0),
    "adamw_001": ("adamw", 0.01),
    "adamw_01": ("adamw", 0.1),
    "l2_000001": ("adam", 0.00001),
    "l2_0001": ("adam", 0.001),
    "l2_01": ("adam", 0.1),
}
MODEL = dict(context=128, width=128, layers=4, heads=4, dropout=0.1)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Block(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, c // self.heads).transpose(1, 2) for a in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(b, t, c)))
        return x + self.drop(self.ff(self.norm2(x)))


class TinyGPT(nn.Module):
    def __init__(self, vocab, context=128, width=128, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[Block(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        return self.head(self.norm(self.blocks(self.token(x) + self.position(positions))))


def decay_groups(model, wd):
    # All linear weight matrices (including the untied output head) decay.
    # Embeddings, LayerNorm and all biases do not decay, for every method.
    decay_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    return [
        {"params": [p for p in model.parameters() if id(p) in decay_ids], "weight_decay": wd},
        {"params": [p for p in model.parameters() if id(p) not in decay_ids], "weight_decay": 0.0},
    ]


def optimizer_for(model, name, lr, device):
    kind, wd = CONFIGS[name]
    cls = torch.optim.AdamW if kind == "adamw" else torch.optim.Adam
    # PyTorch 2.8 Adam's default uses coupled L2. Keep this explicit.
    kwargs = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8, foreach=True)
    if kind == "adam":
        kwargs["decoupled_weight_decay"] = False
    return cls(decay_groups(model, wd), **kwargs)


def load_data(device):
    path = ROOT / "data" / "input.txt"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256, "Unexpected dataset; check provenance."
    text = raw.decode("utf-8")
    n = len(text)
    a, b = int(0.9 * n), int(0.95 * n)
    splits = dict(train=text[:a], val=text[a:b], test=text[b:])
    chars = sorted(set(splits["train"]))
    stoi = {c: i for i, c in enumerate(chars)}
    # Fail explicitly if the held-out text contains an unseen character.
    assert all(set(s).issubset(stoi) for s in splits.values())
    data = {k: torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device) for k, s in splits.items()}
    meta = dict(source="https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare",
                sha256=DATA_SHA256, bytes=len(raw), characters=n, vocab=chars,
                sizes={k: len(v) for k, v in splits.items()}, boundaries=[0, a, b, n],
                test_policy="Final checkpoint only; test unused by learning-rate selection.")
    return data, meta


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, data, device, max_blocks=None):
    model.eval()
    context = MODEL["context"]
    starts = torch.arange(0, len(data) - context, context, device=device)
    if max_blocks and len(starts) > max_blocks:
        # Fixed, equally spaced blocks; independent of train RNG.
        starts = starts[torch.linspace(0, len(starts) - 1, max_blocks, device=device).long()]
    total, count = 0.0, 0
    offset = torch.arange(context, device=device)
    for ix in starts.split(32):
        x = data[ix[:, None] + offset]
        y = data[ix[:, None] + offset + 1]
        with amp_context(device):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum")
        total += loss.item()
        count += y.numel()
    model.train()
    return total / count, count


def lr_at(step, steps, peak):
    warmup = max(10, steps // 20)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_one(name, seed, peak_lr, steps, stage, data, device, batch=32):
    folder = ROOT / "results" / stage / (f"{name}_seed{seed}_lr{peak_lr:g}_steps{steps}")
    if (folder / "summary.json").exists():
        return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    # An interrupted run must not append a new trajectory to stale partial logs.
    (folder / "metrics.jsonl").write_text("", encoding="utf-8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = TinyGPT(len(data["vocab"]), **MODEL).to(device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    opt = optimizer_for(model, name, peak_lr, device)
    generator = torch.Generator(device=device).manual_seed(10_000 + seed)
    context = MODEL["context"]
    offset = torch.arange(context, device=device)
    records = []
    started = time.perf_counter()
    train_seconds = 0.0
    last_grad = 0.0

    def log(step):
        train_ce, _ = evaluate(model, data["train"], device, max_blocks=128)
        val_ce, val_tokens = evaluate(model, data["val"], device)
        norm = math.sqrt(sum(p.detach().float().square().sum().item() for p in model.parameters()))
        row = dict(step=step, train_ce=train_ce, val_ce=val_ce, val_ppl=math.exp(val_ce),
                   param_norm=norm, grad_norm=last_grad, train_seconds=train_seconds,
                   wall_seconds=time.perf_counter() - started, val_tokens=val_tokens)
        records.append(row)
        with (folder / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{stage} {name} seed={seed} lr={peak_lr:g} step={step}/{steps} "
              f"train={train_ce:.4f} val={val_ce:.4f} train_s={train_seconds:.1f}", flush=True)

    log(0)
    sync(device)
    tick = time.perf_counter()
    interval = 100 if stage == "main" else steps
    for step in range(steps):
        ix = torch.randint(0, len(data["train"]) - context, (batch,), generator=generator, device=device)
        x = data["train"][ix[:, None] + offset]
        y = data["train"][ix[:, None] + offset + 1]
        lr = lr_at(step, steps, peak_lr)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with amp_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss in {folder}")
        loss.backward()
        # L2 is inserted internally by Adam AFTER clipping the data gradient.
        # This exactly matches the documented torch.optim.Adam semantics.
        gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if (step + 1) % interval == 0 or step + 1 == steps:
            sync(device)
            train_seconds += time.perf_counter() - tick
            last_grad = float(gn)
            log(step + 1)
            sync(device)
            tick = time.perf_counter()
    final = records[-1]
    test_ce, test_tokens = (evaluate(model, data["test"], device) if stage == "main" else (None, 0))
    summary = dict(config=name, kind=CONFIGS[name][0], weight_decay=CONFIGS[name][1], seed=seed,
                   peak_lr=peak_lr, steps=steps, batch=batch, model=MODEL,
                   parameters=sum(p.numel() for p in model.parameters()), initial_hash=initial_hash,
                   **{k: v for k, v in final.items() if k != "step"},
                   test_ce=test_ce, test_ppl=math.exp(test_ce) if test_ce is not None else None,
                   test_tokens=test_tokens, training_tokens=steps * batch * context,
                   tokens_per_second=steps * batch * context / train_seconds,
                   peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                   wall_total_seconds=time.perf_counter() - started,
                   environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                    platform=platform.platform(), device=str(device),
                                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                    precision="BF16 autocast + FP32 parameters/moments" if device.type == "cuda" else "FP32"))
    if stage == "main":
        torch.save(dict(model=model.state_dict(), model_config=MODEL, vocab=data["vocab"],
                        summary=summary), folder / "final_model.pt")
    write_json(folder / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "main"], required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    arrays, meta = load_data(device)
    data = dict(arrays, vocab=meta["vocab"])
    write_json(ROOT / "results" / "dataset.json", meta)
    if args.stage == "smoke":
        run_one("adamw_01", 0, 1e-3, args.steps or 30, "smoke", data, device)
    elif args.stage == "pilot":
        all_rows = [run_one(name, 0, lr, args.steps or 200, "pilot", data, device)
                    for name in args.configs for lr in [3e-4, 1e-3]]
        choices = {name: min((r for r in all_rows if r["config"] == name), key=lambda r: r["val_ce"])["peak_lr"]
                   for name in args.configs}
        selected_path = ROOT / "results" / "selected_lr.json"
        if selected_path.exists():
            choices = dict(json.loads(selected_path.read_text(encoding="utf-8")), **choices)
        write_json(selected_path, choices)
        print("SELECTED", choices, flush=True)
    else:
        choices = json.loads((ROOT / "results" / "selected_lr.json").read_text(encoding="utf-8"))
        for seed in args.seeds:
            for name in args.configs:
                run_one(name, seed, choices[name], args.steps or 1000, "main", data, device)


if __name__ == "__main__":
    main()
