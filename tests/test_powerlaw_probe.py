import sys

import numpy as np

sys.path.insert(0, ".")

import pytest
from transformers import AutoModelForCausalLM

import powerlaw_probe
from powerlaw_probe import (
    get_layers,
    layer_weight,
    load_tok,
    powerlaw_alpha,
    probe,
    resolve_layer_index,
    spectrum_singulars,
)

MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def setup():
    tok = load_tok(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    return model, tok


def test_arch_and_weight(setup):
    model, tok = setup
    layers = get_layers(model)
    assert len(layers) >= 1
    W = layer_weight(layers[len(layers) // 2])
    assert W.shape[0] > 0


def test_spectrum_sorted(setup):
    model, tok = setup
    W = layer_weight(get_layers(model)[0])
    s = spectrum_singulars(W)
    assert len(s) > 4
    assert (s[:-1] >= s[1:]).all()


def test_alpha_finite(setup):
    model, tok = setup
    W = layer_weight(get_layers(model)[0])
    a = powerlaw_alpha(spectrum_singulars(W))
    assert a is not None
    assert -10.0 < a < 10.0


def test_alpha_uses_original_spectral_ranks():
    # After skipping rank 1, the remaining values are exactly 1/rank.
    singulars = np.array([10.0, 1.0, 0.5, 1.0 / 3, 0.25, 0.2])
    assert powerlaw_alpha(singulars, skip=1) == pytest.approx(1.0)


def test_load_tok_does_not_substitute_another_tokenizer(monkeypatch):
    calls = []

    def fail_to_load(name):
        calls.append(name)
        raise OSError("tokenizer unavailable")

    monkeypatch.setattr(powerlaw_probe, "AutoTokenizer", type("Tokenizer", (), {"from_pretrained": staticmethod(fail_to_load)}))
    with pytest.raises(OSError, match="tokenizer unavailable"):
        load_tok("selected/model")
    assert calls == ["selected/model"]


def test_activation_hook_removed_when_tokenization_fails():
    class Handle:
        removed = False

        def remove(self):
            self.removed = True

    class Layer:
        def register_forward_hook(self, hook):
            self.handle = Handle()
            return self.handle

    class Model:
        def __init__(self):
            self.model = type("ModelBody", (), {"layers": [Layer()]})()

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("tokenization failed")

    model = Model()
    with pytest.raises(RuntimeError, match="tokenization failed"):
        powerlaw_probe.collect_activations(model, Tokenizer(), "prompt", 0)
    assert model.model.layers[0].handle.removed


@pytest.mark.parametrize("requested, expected", [(-1, 2), (0, 0), (3, 3)])
def test_resolve_layer_index_accepts_middle_and_in_range(requested, expected):
    assert resolve_layer_index(requested, 5) == expected


@pytest.mark.parametrize("requested", [-2, 5])
def test_resolve_layer_index_rejects_out_of_range(requested):
    with pytest.raises(ValueError, match="--layer"):
        resolve_layer_index(requested, 5)


def test_probe_report(setup):
    model, tok = setup
    r = probe(model, tok, "Hello world", 0)
    for k in ["layer", "alpha_activations", "alpha_weights", "alpha_gap", "top10_variance_frac_activations"]:
        assert k in r
    assert 0.0 < r["top10_variance_frac_activations"] <= 1.0
    assert r["alpha_activations"] is None or -10.0 < r["alpha_activations"] < 10.0
