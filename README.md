# Demo: Out-of-Distribution Generalization Error Bounds for Successful Prediction of Autoregressive Algorithms
This is the demo of numerical calculation of the upper bound of K-run success probability between the In-distribution and the Bridge Distribution.
In this source, we provide a trained weight of PatchTST which can be run directly to validate the upper bound of KSP under ETTh (K=1, T=96).
Besides, we also probvide the original experimental data of visualization of "UpperBound & T" under ETTh (K=1) for four time series forecasting methods.
## Directory
```text
ICML2026/
├─ Experiment/
│  └─ GeneralizationUpperBound/
│     ├─ Formers/
│     │  ├─ FEDformer/
│     │  │  ├─ data_provider/
│     │  │  │  ├─ data_factory.py
│     │  │  │  └─ data_loader.py
│     │  │  ├─ exp/
│     │  │  │  ├─ exp_basic.py
│     │  │  │  └─ exp_main.py
│     │  │  ├─ layers/
│     │  │  │  ├─ AutoCorrelation.py
│     │  │  │  ├─ Autoformer_EncDec.py
│     │  │  │  ├─ Embed.py
│     │  │  │  ├─ FourierCorrelation.py
│     │  │  │  ├─ MultiWaveletCorrelation.py
│     │  │  │  ├─ SelfAttention_Family.py
│     │  │  │  └─ Transformer_EncDec.py
│     │  │  ├─ models/
│     │  │  │  ├─ Autoformer.py
│     │  │  │  ├─ FEDformer.py
│     │  │  │  ├─ Informer.py
│     │  │  │  └─ Transformer.py
│     │  │  └─ utils/
│     │  │     ├─ masking.py
│     │  │     ├─ metrics.py
│     │  │     ├─ timefeatures.py
│     │  │     └─ tools.py
│     │  └─ Pyraformer/
│     │     ├─ pyraformer/
│     │     │  ├─ Layers.py
│     │     │  ├─ Modules.py
│     │     │  ├─ PAM_TVM.py
│     │     │  ├─ Pyraformer_LR.py
│     │     │  ├─ Pyraformer_SS.py
│     │     │  ├─ SubLayers.py
│     │     │  ├─ embed.py
│     │     │  ├─ graph_attention.py
│     │     │  └─ hierarchical_mm_tvm.py
│     │     ├─ utils/
│     │     │  ├─ timefeatures.py
│     │     │  └─ tools.py
│     │     ├─ data_loader.py
│     │     ├─ long_range_main.py
│     │     ├─ preprocess_elect.py
│     │     ├─ preprocess_flow.py
│     │     ├─ preprocess_wind.py
│     │     ├─ simulate_sin.py
│     │     └─ single_step_main.py
│     ├─ checkpoints/
│     │  └─ ETTh1_96_96_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm256_nh4_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/
│     │     └─ checkpoint.pth
│     ├─ data_provider/
│     │  ├─ data_factory.py
│     │  └─ data_loader.py
│     ├─ dataset/
│     │  └─ ETTh1.csv
│     ├─ exp/
│     │  ├─ exp_basic.py
│     │  └─ exp_main.py
│     ├─ layers/
│     │  ├─ AutoCorrelation.py
│     │  ├─ Autoformer_EncDec.py
│     │  ├─ Embed.py
│     │  ├─ PatchTST_backbone.py
│     │  ├─ PatchTST_layers.py
│     │  ├─ RevIN.py
│     │  ├─ SelfAttention_Family.py
│     │  └─ Transformer_EncDec.py
│     ├─ models/
│     │  ├─ Autoformer.py
│     │  ├─ DLinear.py
│     │  ├─ Informer.py
│     │  ├─ Linear.py
│     │  ├─ NLinear.py
│     │  ├─ PatchTST.py
│     │  ├─ Stat_models.py
│     │  └─ Transformer.py
│     ├─ utils/
│     │  ├─ masking.py
│     │  ├─ metrics.py
│     │  ├─ prob.py
│     │  ├─ timefeatures.py
│     │  └─ tools.py
│     ├─ demo.py
│     └─ requirements.txt
├─ figure_drawing/
│  ├─ ETTh_UpperBound_Visualization.csv
│  └─ figure_drawing.py
└─ figure.png
```
## Start
Install the libraries.
```bash
pip install -r requirements.txt
```

Calculate the upper bound
```bash
python demo.py
```

Draw the figure of UpperBound-T
```bash
python figure_drawing.py
```
