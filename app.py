# -*- coding: utf-8 -*-
import os
import json
from io import BytesIO

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt

from Config import config as train_config
from models import LSTM, CNN, TransformerNet, MLP

st.set_page_config(page_title="Raman Classifier (CV Ensemble)", layout="wide")

# ---------------- Paths (Cloud-safe) ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = APP_DIR  # 默认：把 .pth 放在同目录；也可改成 os.path.join(APP_DIR, "models")
DEFAULT_RAMAN_SHIFT_PATH = os.path.join(APP_DIR, "assets", "Raman_shift.csv")  # 推荐放 repo: assets/Raman_shift.csv


# ================= Utility Functions =================
def list_local_pth(model_dir: str):
    return sorted(
        [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.lower().endswith(".pth")]
    )


def safe_read_csv(file, **kwargs):
    # Streamlit uploader 是 BytesIO-like；pandas 会消耗指针，因此每次尝试前要 seek(0)
    for enc in ["utf-8-sig", "utf-8", "gbk", "ansi", "latin1"]:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            return pd.read_csv(file, encoding=enc, **kwargs)
        except Exception:
            continue
    raise ValueError(f"Unable to read file: {getattr(file, 'name', file)}. Please check encoding format.")


@st.cache_resource(show_spinner=False)
def load_raman_shift(path: str, expected_len: int):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find Raman_shift file: {path}")

    try:
        df = pd.read_csv(path)
        if "Raman_Shift" in df.columns:
            arr = pd.to_numeric(df["Raman_Shift"], errors="coerce").dropna().values
        elif df.shape[1] == 1:
            arr = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
        else:
            best, best_cnt = None, -1
            for c in df.columns:
                vec = pd.to_numeric(df[c], errors="coerce")
                cnt = vec.notna().sum()
                if cnt > best_cnt:
                    best, best_cnt = vec, cnt
            arr = best.dropna().values
    except Exception:
        df = pd.read_csv(path, header=None)
        arr = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values

    if arr.shape[0] < expected_len:
        raise ValueError(
            f"Raman_shift length insufficient: required {expected_len}, but found {arr.shape[0]}"
        )
    return arr[:expected_len]


def snv_per_spectrum(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def to_tensor(x_np: np.ndarray, device: torch.device, dtype: torch.dtype):
    # 训练里你用了 unsqueeze(1)，这里保持一致：(N, 1, D)
    t = torch.as_tensor(x_np, dtype=dtype).unsqueeze(1)
    return t.to(device, non_blocking=True)


@torch.no_grad()
def softmax_probs(model, x_t):
    logits = model(x_t)
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def format_results(prob: np.ndarray, class_names: list, min_conf: float = 0.0):
    df = pd.DataFrame(prob, columns=[f"prob_{c}" for c in class_names])
    pred_idx = prob.argmax(axis=1)
    df.insert(0, "pred_label", [class_names[i] for i in pred_idx])
    max_prob = prob.max(axis=1)
    df.insert(1, "conf_max", max_prob)
    df["flag_low_conf"] = (max_prob < min_conf).astype(int)
    return df


def class_summary(df_pred: pd.DataFrame):
    grp = df_pred.groupby("pred_label")["conf_max"].agg(["count", "mean"]).reset_index()
    grp = grp.rename(columns={"count": "samples", "mean": "avg_conf"})
    return grp.sort_values("samples", ascending=False)


def download_df_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf


# ================= Model Loading =================
def _infer_ckpt_format(ckpt):
    """
    支持两种 ckpt：
    A) {"model_state_dict": ..., "class_names": ..., "input_dim": ..., "num_classes": ...}
    B) 直接是 state_dict（这种你当前 load_model_auto 不支持；这里也尽量兼容）
    """
    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt) and ("class_names" in ckpt):
        return "wrapped"
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # 可能是纯 state_dict
        return "state_dict"
    return "unknown"


@st.cache_resource(show_spinner=False)
def load_ensemble(pth_paths, device_choice: str, use_fp32: bool):
    device = torch.device("cuda" if (device_choice == "cuda" and torch.cuda.is_available()) else "cpu")

    # 先读取第一个 ckpt 决定 input_dim / classes
    first = torch.load(pth_paths[0], map_location="cpu")
    fmt = _infer_ckpt_format(first)
    if fmt != "wrapped":
        raise RuntimeError(
            "Your .pth appears not to contain {'model_state_dict','class_names',...}. "
            "Please save checkpoints with those fields, or adjust loader."
        )

    input_dim = int(first.get("input_dim", getattr(train_config, "INPUT_DIM", 0)))
    class_names = list(first["class_names"])
    num_classes = int(first.get("num_classes", len(class_names)))

    # 选择推理 dtype
    dtype = torch.float32 if use_fp32 else getattr(train_config, "TORCH_DTYPE", torch.float64)

    # 候选架构（你也可以把它做成 UI 选择，但这里自动尝试最稳）
    candidates = [
        ("LSTM", lambda: LSTM(input_dim=input_dim, num_classes=num_classes)),
        ("CNN", lambda: CNN(input_dim=input_dim, num_classes=num_classes)),
        ("Transformer", lambda: TransformerNet(input_dim=input_dim, num_classes=num_classes)),
        ("MLP", lambda: MLP(input_dim=input_dim, num_classes=num_classes)),
    ]

    models = []
    arch_name_used = None

    for ckpt_path in pth_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if _infer_ckpt_format(ckpt) != "wrapped":
            raise RuntimeError(f"Checkpoint format not supported: {os.path.basename(ckpt_path)}")

        # 校验 classes / dim 一致
        cn = list(ckpt["class_names"])
        if cn != class_names:
            raise ValueError(f"class_names mismatch in {os.path.basename(ckpt_path)}")
        dim = int(ckpt.get("input_dim", input_dim))
        if dim != input_dim:
            raise ValueError(f"input_dim mismatch in {os.path.basename(ckpt_path)}")

        loaded = False
        last_err = None
        for arch_name, ctor in candidates:
            try:
                m = ctor()
                m.load_state_dict(ckpt["model_state_dict"], strict=True)
                m = m.to(device)
                m = m.float() if use_fp32 else m.double()
                m.eval()
                models.append(m)
                arch_name_used = arch_name
                loaded = True
                break
            except Exception as e:
                last_err = e
                continue

        if not loaded:
            raise RuntimeError(f"Failed to load model {os.path.basename(ckpt_path)}: {last_err}")

    return models, class_names, input_dim, num_classes, arch_name_used, device, dtype


@torch.no_grad()
def ensemble_predict_proba(models, x_t):
    probs = []
    for m in models:
        probs.append(softmax_probs(m, x_t))
    return np.mean(probs, axis=0)


# ==================== UI ====================
st.title("🧪 Raman Spectra Bacteria Classification · 5-fold Ensemble Inference")

pth_paths = list_local_pth(MODEL_DIR)
if len(pth_paths) == 0:
    st.error("No .pth model files found in app directory.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Inference Settings")

    device_choice = st.selectbox(
        "Device",
        ["cuda", "cpu"],
        index=0 if (torch.cuda.is_available() and getattr(train_config, "CUDA", False)) else 1,
    )

    use_fp32 = st.checkbox("Use float32 for faster inference (may slightly change probabilities)", value=True)
    enable_snv = st.checkbox("Apply SNV (must match training)", value=False)
    min_conf = st.slider("Low-confidence threshold", 0.0, 1.0, 0.6, 0.01)
    has_labels = st.checkbox("CSV last column is true label", value=False)

    st.divider()
    st.subheader("Raman_shift")
    raman_mode = st.radio(
        "Source",
        ["Use repo file (assets/Raman_shift.csv)", "Upload Raman_shift.csv"],
        index=0,
    )
    uploaded_raman = None
    if raman_mode == "Upload Raman_shift.csv":
        uploaded_raman = st.file_uploader("Upload Raman_shift.csv", type=["csv"], accept_multiple_files=False)

st.info("Ensemble checkpoints:\n- " + "\n- ".join([os.path.basename(p) for p in pth_paths]))

# 先加载 ensemble（只加载一次，后面每个文件直接 forward）
with st.spinner("Loading ensemble..."):
    try:
        models, class_names, input_dim, num_classes, arch_name, device, dtype = load_ensemble(
            pth_paths, device_choice, use_fp32
        )
        st.success(f"Loaded {len(models)} models | Arch: {arch_name} | input_dim={input_dim} | device={device}")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

# Raman_shift 读取（用 ckpt input_dim 作为 expected_len，避免 Config 不一致）
axis_arr = None
try:
    if raman_mode == "Use repo file (assets/Raman_shift.csv)":
        axis_arr = load_raman_shift(DEFAULT_RAMAN_SHIFT_PATH, expected_len=input_dim)
        st.sidebar.success(f"Loaded repo Raman_shift: {len(axis_arr)} points")
    else:
        if uploaded_raman is not None:
            # 用上传文件读取：先读成 df 再抽一列
            tmp = safe_read_csv(uploaded_raman)
            if "Raman_Shift" in tmp.columns:
                arr = pd.to_numeric(tmp["Raman_Shift"], errors="coerce").dropna().values
            elif tmp.shape[1] == 1:
                arr = pd.to_numeric(tmp.iloc[:, 0], errors="coerce").dropna().values
            else:
                # 取可用数最多的列
                best, best_cnt = None, -1
                for c in tmp.columns:
                    vec = pd.to_numeric(tmp[c], errors="coerce")
                    cnt = vec.notna().sum()
                    if cnt > best_cnt:
                        best, best_cnt = vec, cnt
                arr = best.dropna().values
            if arr.shape[0] < input_dim:
                raise ValueError(f"Uploaded Raman_shift too short: need {input_dim}, got {arr.shape[0]}")
            axis_arr = arr[:input_dim]
            st.sidebar.success(f"Loaded uploaded Raman_shift: {len(axis_arr)} points")
        else:
            st.sidebar.warning("Please upload Raman_shift.csv (or switch to repo file).")
except Exception as e:
    st.sidebar.error(f"Raman_shift error: {e}")

# ==================== Upload and Predict ====================
st.write("上传一个或多个 CSV：每行一个样本，列为特征（不含标签列）。如果勾选“CSV last column is true label”，则最后一列会作为真实标签。")
data_files = st.file_uploader("Upload .csv file(s)", type=["csv"], accept_multiple_files=True)

with st.expander("Optional: Plot settings", expanded=False):
    show_plots = st.checkbox("Show class-wise mean spectra plots", value=True)
    top_n_plot = st.slider("Plot top-N classes by sample count", 1, min(10, num_classes), min(5, num_classes))

if st.button("Start Prediction", type="primary"):
    if not data_files:
        st.error("Please upload at least one CSV file.")
        st.stop()

    if axis_arr is None:
        st.error("Raman_shift is not loaded yet. Please fix Raman_shift first.")
        st.stop()

    status = st.status("Running inference...", expanded=True)
    progress = st.progress(0)

    all_results = []
    total_files = len(data_files)

    for fi, file in enumerate(data_files, start=1):
        status.write(f"📄 Processing: {file.name} ({fi}/{total_files})")

        try:
            raw_df = safe_read_csv(file, header=None)
        except Exception as e:
            st.error(f"{file.name} failed to read: {e}")
            continue

        X = raw_df.iloc[:, :-1].values if has_labels else raw_df.values
        y_true_raw = raw_df.iloc[:, -1].astype(str).values if has_labels else None

        X = X.astype(np.float64)
        if enable_snv:
            X = snv_per_spectrum(X)

        if X.shape[1] != len(axis_arr):
            st.error(
                f"{file.name} feature count mismatch: X has {X.shape[1]} features, "
                f"but Raman_shift has {len(axis_arr)} points."
            )
            continue

        # 推理
        x_t = to_tensor(X, device=device, dtype=dtype)
        probs_ens = ensemble_predict_proba(models, x_t)

        df_pred = format_results(probs_ens, class_names, min_conf=min_conf)
        df_pred["source_file"] = file.name

        if has_labels:
            df_pred["true_label"] = y_true_raw
            df_pred["is_correct"] = (df_pred["pred_label"] == df_pred["true_label"]).astype(int)

        all_results.append(df_pred)

        # 展示：每个文件汇总
        st.subheader(f"✅ File: {file.name}")
        summary_df = class_summary(df_pred)
        st.write("Class Summary:")
        st.dataframe(summary_df, use_container_width=True)
        st.bar_chart(summary_df.set_index("pred_label")["samples"])

        if has_labels:
            acc = float(df_pred["is_correct"].mean()) if len(df_pred) else 0.0
            st.metric("Accuracy (this file)", f"{acc:.3f}")

        # 可选：每类平均谱图（默认 Top-N 类）
        if show_plots:
            top_classes = summary_df["pred_label"].head(top_n_plot).tolist()
            st.write("Average Spectrum per Class (Top-confidence samples, Top-N classes)")
            for cls in top_classes:
                mask = df_pred["pred_label"] == cls
                idx_cls = np.where(mask.values)[0]
                if len(idx_cls) == 0:
                    continue

                conf = df_pred.loc[idx_cls, "conf_max"].values
                order = np.argsort(-conf)
                top_idx = idx_cls[order[: min(5, len(order))]]

                X_top = X[top_idx, :]
                mean_s, std_s = X_top.mean(axis=0), X_top.std(axis=0)

                fig, ax = plt.subplots()
                ax.plot(axis_arr, mean_s, label=f"{cls} - mean")
                ax.fill_between(axis_arr, mean_s - std_s, mean_s + std_s, alpha=0.2)
                ax.set_xlabel("Raman Shift")
                ax.set_ylabel("Intensity")
                ax.set_title(f"{cls} Mean Spectrum ±1 std (top {len(top_idx)} samples)")
                ax.legend()

                st.pyplot(fig, clear_figure=True)
                png_buf = fig_to_bytes(fig)
                st.download_button(
                    f"⬇️ Download {cls} Mean Spectrum",
                    data=png_buf,
                    file_name=f"{os.path.splitext(file.name)[0]}_{cls}_meanstd.png",
                    mime="image/png",
                )
                plt.close(fig)

        progress.progress(int(100 * fi / total_files))

    if not all_results:
        status.update(label="No valid results.", state="error")
        st.stop()

    df_pred_all = pd.concat(all_results, ignore_index=True)

    status.update(label="✅ Inference completed.", state="complete")
    st.success(f"✅ Batch inference completed, total {len(df_pred_all)} samples.")

    st.subheader("📊 Overall Class Summary")
    summary_all = class_summary(df_pred_all)
    st.dataframe(summary_all, use_container_width=True)
    st.bar_chart(summary_all.set_index("pred_label")["samples"])

    if has_labels and "is_correct" in df_pred_all.columns:
        overall_acc = float(df_pred_all["is_correct"].mean()) if len(df_pred_all) else 0.0
        st.metric("Overall Accuracy", f"{overall_acc:.3f}")

    download_df_button(df_pred_all, "all_predictions.csv", "⬇️ Download Overall Predictions")

st.caption("© Batch Prediction Visualization · Ensemble Averaging · Cloud-safe Raman_shift loading")
