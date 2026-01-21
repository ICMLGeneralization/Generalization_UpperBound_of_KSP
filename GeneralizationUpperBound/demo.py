import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from exp.exp_main import Exp_Main
from utils.prob import HittingProbUpperBoundCalculator

def build_default_config() -> dict:
    cfg = {
        "random_seed": 2021,
        "is_training": 1,
        "model_id": "test",
        "model": "Autoformer",
        "data": "ETTm1",
        "root_path": os.path.join(os.path.dirname(__file__), "dataset"),
        "data_path": "ETTh1.csv",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "checkpoints": os.path.join(os.path.dirname(__file__), "checkpoints"),
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 96,
        "fc_dropout": 0.05,
        "head_dropout": 0.0,
        "patch_len": 16,
        "stride": 8,
        "padding_patch": "end",
        "revin": 1,
        "affine": 0,
        "subtract_last": 0,
        "decomposition": 0,
        "kernel_size": 25,
        "individual": 0,
        "embed_type": 0,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 2048,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "dropout": 0.05,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": False,
        "do_predict": False,
        "num_workers": 1,
        "itr": 2,
        "train_epochs": 100,
        "batch_size": 128,
        "patience": 100,
        "learning_rate": 0.0001,
        "des": "test",
        "loss": "mse",
        "lradj": "type3",
        "pct_start": 0.3,
        "use_amp": False,
        "use_gpu": True,
        "gpu": 0,
        "use_multi_gpu": False,
        "devices": "0,1,2,3",
        "test_flop": False,
    }
    return cfg


def validate_config(cfg: dict) -> None:
    required_keys = [
        "is_training",
        "model_id",
        "model",
        "data",
        "root_path",
        "data_path",
        "features",
        "seq_len",
        "label_len",
        "pred_len",
        "enc_in",
        "dec_in",
        "c_out",
        "d_model",
        "n_heads",
        "e_layers",
        "d_layers",
        "d_ff",
        "factor",
        "embed",
        "distil",
        "des",
        "itr",
        "train_epochs",
        "batch_size",
        "learning_rate",
        "use_gpu",
        "gpu",
        "use_multi_gpu",
        "devices",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")


def cfg_to_args(cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(**cfg)


def fix_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def normalize_device_config(args: SimpleNamespace) -> None:
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = str(args.devices).replace(" ", "")
        device_ids = [int(x) for x in args.devices.split(",") if x != ""]
        args.device_ids = device_ids
        args.gpu = device_ids[0] if device_ids else int(args.gpu)


def build_setting(args: SimpleNamespace, ii: int) -> str:
    return (
        "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}"
    ).format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii,
    )


def main() -> None:
    cfg = build_default_config()

    cfg.update(
        {
            "is_training": 1,
            "model_id": "ETTh1_96_96",
            "model": "PatchTST",
            "data": "ETTh1",
            "root_path": os.path.join(os.path.dirname(__file__), "dataset"),
            "data_path": "ETTh1.csv",
            "features": "M",
            "seq_len": 96, 
            "label_len": 48,
            "pred_len": 96,
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "itr": 1, 
            "train_epochs": 2000,  
            "des": "Exp",
            "num_workers": 0,
            "batch_size": 384,
            "patience": 100,
            "learning_rate": 0.00001,
        }
    )

    validate_config(cfg)
    args = cfg_to_args(cfg)

    fix_random_seed(int(args.random_seed))
    normalize_device_config(args)

    print("Args in experiment:")
    print(vars(args))

    Exp = Exp_Main

    if args.is_training:
        for ii in range(int(args.itr)):
            setting = build_setting(args, ii)
            exp = Exp(args)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            a_preds, a_trues = exp.train(setting)

            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            t_preds, t_trues = exp.test(setting)

            if getattr(args, "do_predict", False):
                print(f">>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.predict(setting, True)

            torch.cuda.empty_cache()
        return a_preds, a_trues, t_preds, t_trues
    else:
        ii = 0
        setting = build_setting(args, ii)
        exp = Exp(args)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        preds, tures = exp.test(setting, test=1)
        torch.cuda.empty_cache()
        return np.zeros_like(preds), np.zeros_like(tures), preds, tures


if __name__ == "__main__":
    item = main()
    if item is not None:
        a_preds, a_trues, t_preds, t_trues = item
        a_preds = torch.from_numpy(a_preds).to(torch.float16)
        a_trues = torch.from_numpy(a_trues).to(torch.float16)
        t_preds = torch.from_numpy(t_preds).to(torch.float16)
        t_trues = torch.from_numpy(t_trues).to(torch.float16)

        h = HittingProbUpperBoundCalculator(dataset="ETTh_1K")
        a_hitting_prob = h.cal_hitting_prob(a_preds, a_trues)
        t_hitting_prob = h.cal_hitting_prob(t_preds, t_trues)

        theo_bound, expre_bound = h.cal_theoretical_upper_bound_pair(0.5, a_hitting_prob, t_hitting_prob)
        print(f"[theoretical] upper bound: {theo_bound.item()}, [experimental] upper bound: {expre_bound}")
        
        


