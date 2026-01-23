# Numerical Calculation Demo (Anonymous Release)

This repository provides a numerical demonstration of the proposed **upper bound on the (K)-run success probability (KSP)** between the *in-distribution* (ID) and the *bridge distribution* (BD).

The demo includes:

* A **pretrained PatchTST model** that can be directly used to validate the theoretical upper bound on KSP under the **ETTh1** dataset with (K=1) and prediction horizon (T=96).
* The **original experimental data** used to visualize the relationship between the upper bound and the prediction length (T) (“UpperBound–(T)”) under ETTh1, covering **four time-series forecasting models**.

This code is intended for **numerical verification and visualization**, rather than large-scale benchmarking.

---

## 🗂️ Directory Structure

```text
ICML2026/
├─ Experiment/
│  └─ GeneralizationUpperBound/
│     ├─ Formers/
│     │  ├─ FEDformer/
│     │  │  ├─ data_provider/
│     │  │  ├─ exp/
│     │  │  ├─ layers/
│     │  │  ├─ models/
│     │  │  └─ utils/
│     │  └─ Pyraformer/
│     │     ├─ pyraformer/
│     │     ├─ utils/
│     │     ├─ data_loader.py
│     │     ├─ long_range_main.py
│     │     ├─ preprocess_*.py
│     │     └─ single_step_main.py
│     ├─ checkpoints/
│     │  └─ ETTh1_96_96_PatchTST_*/
│     │     └─ checkpoint.pth
│     ├─ data_provider/
│     ├─ dataset/
│     │  └─ ETTh1.csv
│     ├─ exp/
│     ├─ layers/
│     ├─ models/
│     ├─ utils/
│     ├─ demo.py
│     └─ requirements.txt
├─ figure_drawing/
│  ├─ ETTh_UpperBound_Visualization.csv
│  └─ figure_drawing.py
└─ figure.png
```

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Compute the Upper Bound

```bash
python demo.py
```

Training can be **time-consuming**.
For quick validation using the provided pretrained weights, set:

```python
is_training = 0
```

in `demo.py` (line 156), and then run the script.

The repository already includes trained PatchTST weights under ETTh1 with:
$$K=1,\quad T=96,\quad a=0.0,\quad \varepsilon = 1\sigma.$$

**⚠️Important:**
If you train the model from scratch (`is_training = 1`), you must set `is_training = 0` **after training** and rerun `demo.py` to obtain the final numerical results.

<p align="center">
  <img width="1133" height="699" src="https://github.com/user-attachments/assets/81560b3c-0f28-48db-9e6f-b481b13db4fe" />
</p>

---

### Visualize UpperBound–(T)

```bash
python figure_drawing.py
```

This script reproduces the **UpperBound–(T)** curves based on the provided experimental data.







