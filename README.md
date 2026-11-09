# ⚖️ POWERLAW-PROBE

[![CI](https://github.com/kapildhinwa2010-dev/POWERLAW-PROBE/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/POWERLAW-PROBE/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/POWERLAW-PROBE/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/POWERLAW-PROBE/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Do a model's activations and weights share a shape?

One probe, two spectra, one number that matters: the **alpha gap**.

- 🧠 Run a prompt through the model, collect a live activation matrix
- ⚖️ Take the same layer's weight matrix
- 📉 Fit a power-law exponent (alpha) to each singular value spectrum
- 🦋 Compare: if the alphas disagree, the weights are a **compressed picture**
  of the behavior, not the behavior itself

## 🔍 What you typically see

- 📈 Activations often concentrate: steep power law, top-10 singular
  directions capture nearly everything
- 📉 Weights often stay flat: energy spread across many directions
- ⚖️ The gap between the two alphas is a fingerprint of the layer

## ⚡ Install

GPU (default):

    pip install torch
    pip install -r requirements.txt

<details>
<summary>🖥️ CPU-only variant</summary>

    pip install --index-url https://download.pytorch.org/whl/cpu torch
    pip install -r requirements.txt

</details>

## 💻 Usage

    python powerlaw_probe.py --model Qwen/Qwen2.5-0.5B-Instruct

Output: console JSON + `powerlaw_report.json`.

## 🧪 Tests

    pytest -q

<details>
<summary>🔬 Honest limitations</summary>

- A log-log linear fit is the simplest power-law estimator; tails and
  finite-size effects can shift alpha. Compare layers of the same model,
  not models of different sizes.
- The alpha gap is a descriptive fingerprint, not a causal claim about
  compression or behavior.

</details>

## ⚖️ License

MIT - see [LICENSE](LICENSE).
