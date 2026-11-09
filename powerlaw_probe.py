"""POWERLAW-PROBE: do a model's activations and weights share a shape?

Compute the singular value spectrum of (a) a live activation matrix and
(b) the layer's weight matrix, then fit a power-law exponent alpha to each
(log-log linear fit). The alpha gap is a fingerprint:

- activations often live near a steeper power law,
- weights often sit near a much flatter one.

If the two alphas disagree, the weights are a compressed picture of the
behavior, not the behavior itself.

GPU-first, CPU-supported. No training. Bounded evaluation.
"""

__version__ = "0.1.0"

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("unsupported architecture")


def load_tok(name):
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2")


def layer_weight(layer):
    """A representative weight matrix of the layer, any dtype -> float32."""
    for attr in ("mlp.gate_proj", "mlp.c_fc", "mlp.fc_in", "c_fc"):
        node = layer
        ok = True
        for part in attr.split("."):
            if hasattr(node, part):
                node = getattr(node, part)
            else:
                ok = False
                break
        if ok and hasattr(node, "weight"):
            return node.weight.detach().float()
    raise ValueError("no weight found for layer")


@torch.no_grad()
def collect_activations(model, tok, prompt, layer_idx, max_len=64):
    acts = []

    def hook(mod, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        acts.append(y.reshape(-1, y.shape[-1]).float().cpu())

    handle = get_layers(model)[layer_idx].register_forward_hook(hook)
    ids = tok(prompt, return_tensors="pt").input_ids[:, :max_len]
    model(ids)
    handle.remove()
    return torch.cat(acts, dim=0)


def spectrum_singulars(M):
    """Singular values sorted descending, as numpy."""
    M = M.detach().float().cpu() if isinstance(M, torch.Tensor) else np.asarray(M, dtype=np.float32)
    Mt = torch.as_tensor(M)
    S = torch.linalg.svdvals(Mt).numpy()
    return np.sort(S)[::-1]


def powerlaw_alpha(singulars, skip=1):
    """Fit log(s) vs log(rank) on the tail after the top 'skip' values."""
    s = singulars[skip:]
    s = s[s > 1e-12]
    if len(s) < 4:
        return None
    ranks = np.arange(1, len(s) + 1)
    slope = np.polyfit(np.log(ranks), np.log(s), 1)[0]
    return float(-slope)


def probe(model, tok, prompt, layer_idx):
    A = collect_activations(model, tok, prompt, layer_idx)
    W = layer_weight(get_layers(model)[layer_idx])
    s_act = spectrum_singulars(A)
    s_w = spectrum_singulars(W)
    a_act = powerlaw_alpha(s_act)
    a_w = powerlaw_alpha(s_w)
    top10_act = float((s_act[:10] ** 2).sum() / ((s_act ** 2).sum() + 1e-12))
    return {
        "layer": layer_idx,
        "alpha_activations": a_act,
        "alpha_weights": a_w,
        "alpha_gap": None if (a_act is None or a_w is None) else round(a_act - a_w, 4),
        "top10_variance_frac_activations": round(top10_act, 4),
        "n_activation_rows": int(A.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser(prog="powerlaw-probe", description="Power-law exponents of activations vs weights.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="Tokyo is the capital of Japan, a country in East Asia.")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = middle layer")
    ap.add_argument("--output", default="powerlaw_report.json")
    a = ap.parse_args()

    tok = load_tok(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()

    n_layers = len(get_layers(model))
    idx = n_layers // 2 if a.layer == -1 else a.layer
    report = {"model": a.model, "n_layers": n_layers, "probe": probe(model, tok, a.prompt, idx)}

    print(json.dumps(report, indent=2))
    with open(a.output, "w") as f:
        json.dump(report, f, indent=2)
    print("saved: " + a.output)


if __name__ == "__main__":
    main()
