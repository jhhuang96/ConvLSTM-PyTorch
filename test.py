# test.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# ---- Fix: Windows OpenMP duplicated runtime (libiomp5md.dll) ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import (
    convlstm_encoder_params, convlstm_decoder_params,
    convgru_encoder_params,  convgru_decoder_params
)
from data.mm import MovingMNIST

import matplotlib
matplotlib.use("Agg")  # 只保存图，不弹窗（更稳）
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_net(model_type: str) -> nn.Module:
    model_type = model_type.lower().strip()
    if model_type == "convlstm":
        encoder_params = convlstm_encoder_params
        decoder_params = convlstm_decoder_params
    elif model_type == "convgru":
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params
    else:
        raise ValueError(f"Unknown model_type={model_type}")

    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)
    return net


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_to_net(net: nn.Module, ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)

    # EarlyStopping 保存的是 dict: {'epoch','state_dict','optimizer'}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # 兜底：有时直接保存 state_dict
        state_dict = ckpt
        ckpt = {"epoch": None, "state_dict": state_dict}
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    state_dict = strip_module_prefix(state_dict)
    net.load_state_dict(state_dict, strict=True)
    return ckpt


def try_load_auto(ckpt_path: str, device: torch.device):
    # auto: 先 convlstm 再 convgru
    errors = {}
    for mt in ["convlstm", "convgru"]:
        try:
            net = build_net(mt).to(device)
            ckpt = load_checkpoint_to_net(net, ckpt_path, device)
            return mt, net, ckpt
        except Exception as e:
            errors[mt] = repr(e)
    raise RuntimeError(f"Auto load failed. Errors: {errors}")


def save_compare_figure(
    save_path: str,
    inputs: torch.Tensor,   # [Tin, C, H, W]
    gts: torch.Tensor,      # [Tout, C, H, W]
    preds: torch.Tensor,    # [Tout, C, H, W]
    vmin: float = 0.0,
    vmax: float = 1.0
):
    """
    生成 3 行对比图（Input / GT / Pred），列数 = Tin + Tout

    布局：
    1) Input：前 Tin 列显示 input，后 Tout 列显示空白
    2) Ground Truth：前 Tin 列显示 input，后 Tout 列显示未来真值
    3) Prediction：前 Tin 列显示 input，后 Tout 列显示模型预测
    """
    Tin = inputs.shape[0]
    Tout = gts.shape[0]
    total = Tin + Tout

    inp_np = inputs.detach().cpu().numpy()
    gt_np = gts.detach().cpu().numpy()
    pr_np = preds.detach().cpu().numpy()

    # 空白帧：与 input 同形状，数值为 0（黑底）
    blank = np.zeros_like(inp_np[:1])                      # [1,C,H,W]
    blanks = np.repeat(blank, repeats=Tout, axis=0)        # [Tout,C,H,W]

    row_inp = np.concatenate([inp_np, blanks], axis=0)     # [total,C,H,W]
    row_gt  = np.concatenate([inp_np, gt_np], axis=0)      # [total,C,H,W]
    row_pr  = np.concatenate([inp_np, pr_np], axis=0)      # [total,C,H,W]

    fig_h = 3.2
    fig_w = max(10.0, total * 0.8)
    fig, axes = plt.subplots(3, total, figsize=(fig_w, fig_h), squeeze=False)

    titles = ["Input", "Ground Truth", "Prediction"]
    rows = [row_inp, row_gt, row_pr]

    for r in range(3):
        for t in range(total):
            ax = axes[r, t]
            img = rows[r][t, 0]  # C=1 -> 取第0通道
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:  # 顶部标注时间索引
                if t < Tin:
                    ax.set_title(f"in{t}", fontsize=8)
                else:
                    ax.set_title(f"t+{t-Tin}", fontsize=8)

        axes[r, 0].set_ylabel(titles[r], fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=r"./save_model/2020-03-09T00-00-00/checkpoint_0_0.009662.pth.tar",
                        help="path to checkpoint_*.pth.tar")
    parser.add_argument("--data_root", type=str, default="data/", help="data root folder")
    parser.add_argument("--frames_input", type=int, default=10)
    parser.add_argument("--frames_output", type=int, default=10)
    parser.add_argument("--num_objects", type=int, default=3, help="MovingMNIST num_objects=[N]")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=8, help="how many samples to visualize")
    parser.add_argument("--save_dir", type=str, default="./test_results")
    parser.add_argument("--seed", type=int, default=1996)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "convlstm", "convgru"],
                        help="model cell type to build")
    args = parser.parse_args()

    set_seed(args.seed)

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # dataset / loader
    ds = MovingMNIST(
        is_train=False,
        root=args.data_root,
        n_frames_input=args.frames_input,
        n_frames_output=args.frames_output,
        num_objects=[args.num_objects],
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    # build + load
    if args.model_type == "auto":
        mt, net, ckpt = try_load_auto(args.checkpoint, device)
    else:
        mt = args.model_type
        net = build_net(mt).to(device)
        ckpt = load_checkpoint_to_net(net, args.checkpoint, device)

    net.eval()

    print(f"[Info] device={device}")
    print(f"[Info] model_type={mt}")
    print(f"[Info] checkpoint={args.checkpoint}")
    print(f"[Info] checkpoint_epoch={ckpt.get('epoch', None)}")

    os.makedirs(args.save_dir, exist_ok=True)

    saved = 0
    for batch_i, (idx, targetVar, inputVar, _, _) in enumerate(loader):
        # input: [B,Tin,C,H,W], target: [B,Tout,C,H,W]
        inputs = inputVar.to(device)
        gts = targetVar.to(device)

        preds = net(inputs)

        # metrics
        mse = F.mse_loss(preds, gts).item()
        per_t = ((preds - gts) ** 2).mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()  # [Tout]

        # save each sample in batch
        B = inputs.shape[0]
        for b in range(B):
            sample_idx = int(idx[b].item()) if torch.is_tensor(idx) else int(idx[b])
            out_png = os.path.join(args.save_dir, f"sample_{sample_idx:05d}.png")

            save_compare_figure(
                out_png,
                inputs=inputs[b],   # [Tin,C,H,W]
                gts=gts[b],         # [Tout,C,H,W]
                preds=preds[b],     # [Tout,C,H,W]
                vmin=0.0, vmax=1.0
            )

            per_t_str = ", ".join([f"{x:.6f}" for x in per_t])
            print(f"[Sample {sample_idx:05d}] MSE={mse:.8f} | per_t=[{per_t_str}] | saved: {out_png}")

            saved += 1
            if saved >= args.num_samples:
                print(f"[Done] saved {saved} samples to: {args.save_dir}")
                return

    print(f"[Done] dataset exhausted, saved {saved} samples to: {args.save_dir}")


if __name__ == "__main__":
    main()
