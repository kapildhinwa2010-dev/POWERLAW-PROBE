import sys

sys.path.insert(0, ".")

import pytest
from transformers import AutoModelForCausalLM

from powerlaw_probe import get_layers, layer_weight, load_tok, powerlaw_alpha, probe, spectrum_singulars

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


def test_probe_report(setup):
    model, tok = setup
    r = probe(model, tok, "Hello world", 0)
    for k in ["layer", "alpha_activations", "alpha_weights", "alpha_gap", "top10_variance_frac_activations"]:
        assert k in r
    assert 0.0 < r["top10_variance_frac_activations"] <= 1.0
    assert r["alpha_activations"] is None or -10.0 < r["alpha_activations"] < 10.0
