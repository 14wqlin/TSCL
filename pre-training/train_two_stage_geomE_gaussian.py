# -*- coding: utf-8 -*-
"""
train_two_stage_geomE_gaussian_stage2rxn.py

在 train_two_stage_geomE_gaussian.py 基础上，只修改 Stage2 的损失形式，
其余功能（Stage1、Geom-only 分支、打印/保存/作图）全部保留。

新的 Stage2 总损失：
    L_stage2 = lambda_stage2_multilevel * (lambda_rxn_stage1 * L_rxn
                                           + lambda_coarse7_stage1 * L_coarse7)
               + lambda_stage2_geom * L_geomE

也就是说：
  - Stage2 中继续使用与 Stage1 相同的 reaction3 + coarse7 多级 SupCon 结构；
  - lambda_rxn_stage1, lambda_coarse7_stage1 与 Stage1 相同（表示二者内部权重）；
  - lambda_stage2_multilevel 控制 “多级 SupCon” 在 Stage2 中整体的强度；
  - lambda_stage2_geom 控制 geomE 连续对比损失的强度。

这样可以最大程度地继承 Stage1 的 3 个 reaction 大簇 + 7 个 coarse 小簇结构，
同时在其上叠加 geomE 细化，从而减少 direct 反应簇被严重拉裂的风险。

此外，将默认 max_stage2_epochs 从 200 提升至 300，以便 Stage2 有更
充分的收敛空间（你可以根据后续势垒回归表现再调小）。
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import gridspec  # 用于自定义 colorbar 位置

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10
SCATTER_SIZE = 10


# ======================================================================
# 0. 基础工具函数 （与原脚本完全相同）
# ======================================================================

def set_seed(seed=42):
    """固定随机种子，保证实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_struct_embeddings_and_reactions(out_dir="embeddings"):
    """
    从指定目录加载结构 embedding、substrate、reaction_type 等。

    必需文件：
      - IS_embeddings.npy, FS_embeddings.npy
      - substrates.npy, reaction_types.npy
      - IS_names.npy, FS_names.npy
    Δgeom_4 和 ΔE_mlp 在后面单独加载（因为你做了筛选）。
    """
    IS_embeddings = np.load(os.path.join(out_dir, "IS_embeddings.npy"))
    FS_embeddings = np.load(os.path.join(out_dir, "FS_embeddings.npy"))
    substrates = np.load(os.path.join(out_dir, "substrates.npy"))
    reaction_types = np.load(os.path.join(out_dir, "reaction_types.npy"))
    IS_names = np.load(os.path.join(out_dir, "IS_names.npy"))
    FS_names = np.load(os.path.join(out_dir, "FS_names.npy"))
    return IS_embeddings, FS_embeddings, substrates, reaction_types, IS_names, FS_names


def load_geom4_and_deltaE(out_dir="embeddings"):
    """
    加载最新的几何特征和 ΔE_mlp：

      - geom_delta_080_4_fea.npy : (N,4)，你根据 Pearson 相关性筛选后的 Δgeom；
      - deltaE_mlp.npy           : (N,)，IS -> FS 的 ΔE_mlp。
    """
    geom4 = np.load(os.path.join(out_dir, "geom_delta_080_4_fea.npy"))
    deltaE_path = os.path.join(out_dir, "deltaE_mlp.npy")
    if os.path.exists(deltaE_path):
        deltaE = np.load(deltaE_path)
        print(f"[Geom4] 读取 Δgeom_4, 形状={geom4.shape}")
        print(f"[ΔE]    读取 deltaE_mlp, 形状={deltaE.shape}")
    else:
        deltaE = None
        print("[警告] 未找到 deltaE_mlp.npy，将在几何对比学习中不使用 ΔE。")
    return geom4, deltaE


# ======================================================================
# 1. ReactionEncoder 与 embedding 计算（与原脚本相同）
# ======================================================================

class ReactionEncoder(nn.Module):
    """
    反应级编码器：输入 (h_IS, h_FS)，输出归一化的反应 embedding z。

    使用 "concat_diff" 模式：
      x = [h_IS, h_FS, h_FS - h_IS]
      -> 2 层 MLP -> proj_dim (默认 128) -> L2 归一化。
    """

    def __init__(self, struct_dim=64, proj_dim=128, mode="concat_diff"):
        super().__init__()
        assert mode in ["concat", "concat_diff"]
        self.mode = mode
        if mode == "concat":
            in_dim = struct_dim * 2
        else:
            in_dim = struct_dim * 3
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, h_IS, h_FS):
        if self.mode == "concat":
            x = torch.cat([h_IS, h_FS], dim=-1)
        else:
            x = torch.cat([h_IS, h_FS, h_FS - h_IS], dim=-1)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        return x


def compute_reaction_embeddings(IS_embeddings, FS_embeddings, encoder, device="cpu"):
    """用 encoder 对全部 (IS,FS) 计算 reaction embedding Z（numpy）。"""
    encoder.eval()
    with torch.no_grad():
        t_IS = torch.from_numpy(IS_embeddings).float().to(device)
        t_FS = torch.from_numpy(FS_embeddings).float().to(device)
        z = encoder(t_IS, t_FS)
        z = F.normalize(z, dim=-1)
    return z.cpu().numpy()


# ======================================================================
# 2. coarse7 & reaction3 标签定义（与原脚本相同）
# ======================================================================

def coarse7_label_of_idx(i, substrates, reaction_types) -> int:
    """
    根据 (substrate, reaction_type) 定义 7 类 coarse 标签：

      0: zeolite / C-H_activation
      1: zeolite / C-O_coupling
      2: alloy   / C-H_activation
      3: alloy   / C-O_coupling
      4: oxide        / C-H_activation
      5: MN4-graphene / CH4_CH3OH_direct_oxidation
      6: MOF          / CH4_CH3OH_direct_oxidation
     -1: 其他组合（不参与 coarse7 监督）
    """
    sub = str(substrates[i]).strip()
    rt = str(reaction_types[i]).strip()

    if sub == "zeolite" and rt == "C-H_activation":
        return 0
    if sub == "zeolite" and rt == "C-O_coupling":
        return 1
    if sub == "alloy" and rt == "C-H_activation":
        return 2
    if sub == "alloy" and rt == "C-O_coupling":
        return 3
    if sub == "oxide" and rt == "C-H_activation":
        return 4
    if sub == "MN4-graphene" and rt == "CH4_CH3OH_direct_oxidation":
        return 5
    if sub == "MOF" and rt == "CH4_CH3OH_direct_oxidation":
        return 6
    return -1


def reaction3_label_of_idx(i, reaction_types) -> int:
    """
    只按 reaction_type 划分 3 大类：

      0: C-H_activation
      1: C-O_coupling
      2: CH4_CH3OH_direct_oxidation
    """
    rt = str(reaction_types[i]).strip()
    if rt == "C-H_activation":
        return 0
    if rt == "C-O_coupling":
        return 1
    if rt == "CH4_CH3OH_direct_oxidation":
        return 2
    return -1


def build_coarse7_and_rxn_groups(subs, rtypes):
    """
    为所有样本构建：
      - coarse7_labels: (N,) ∈ {-1,0..6}
      - reaction3_labels: (N,) ∈ {-1,0..2}
      - coarse7_groups: dict[c] -> idx_list
      - reaction3_groups: dict[r] -> idx_list
    """
    N = len(subs)
    coarse7 = np.zeros(N, dtype=int)
    rxn3 = np.zeros(N, dtype=int)
    for i in range(N):
        coarse7[i] = coarse7_label_of_idx(i, subs, rtypes)
        rxn3[i] = reaction3_label_of_idx(i, rtypes)

    coarse7_groups = {}
    rxn3_groups = {}
    for i in range(N):
        c = coarse7[i]
        r = rxn3[i]
        if c >= 0:
            coarse7_groups.setdefault(c, []).append(i)
        if r >= 0:
            rxn3_groups.setdefault(r, []).append(i)

    for c in list(coarse7_groups.keys()):
        coarse7_groups[c] = np.array(coarse7_groups[c], dtype=int)
    for r in list(rxn3_groups.keys()):
        rxn3_groups[r] = np.array(rxn3_groups[r], dtype=int)

    print("[Coarse7 分组统计]")
    for c in sorted(coarse7_groups.keys()):
        print(f"  coarse7 {c}: num_samples = {len(coarse7_groups[c])}")
    num_ignored_c = np.sum(coarse7 < 0)
    if num_ignored_c > 0:
        print(f"  coarse7=-1 的样本数: {num_ignored_c}")

    print("[Reaction3 分组统计]")
    for r in sorted(rxn3_groups.keys()):
        print(f"  reaction3 {r}: num_samples = {len(rxn3_groups[r])}")
    num_ignored_r = np.sum(rxn3 < 0)
    if num_ignored_r > 0:
        print(f"  reaction3=-1 的样本数: {num_ignored_r}")

    return coarse7, rxn3, coarse7_groups, rxn3_groups


# ======================================================================
# 3. SupCon 损失（离散 / 连续） 与 采样函数
# ======================================================================

def supcon_loss_discrete_labels(
    z_batch: torch.Tensor,
    batch_indices: np.ndarray,
    global_labels: np.ndarray,
    label_func,
    temperature: float = 0.1,
    min_pos_per_anchor: int = 8,
    max_pos_per_anchor: int = 32,
):
    """
    通用的离散标签 SupCon 损失，用于：
      - Stage1 coarse7（材料+反应 联合标签）
      - Stage1 reaction3（仅反应类型）
      - Stage2 中再次使用 reaction3/coarse7（多级先验）
    """
    device = z_batch.device
    B = z_batch.shape[0]

    labels = np.array([label_func(int(i)) for i in batch_indices], dtype=int)
    labels_t = torch.from_numpy(labels).to(device)

    sim = torch.matmul(z_batch, z_batch.T) / temperature

    losses = []
    total_anchors = 0
    total_pos_pairs = 0
    total_neg_pairs = 0

    for i in range(B):
        lab_i = labels[i]
        if lab_i < 0:
            continue

        mask_pos = (labels_t == lab_i)
        mask_pos[i] = False
        pos_idx = torch.nonzero(mask_pos).squeeze(-1)
        num_pos = pos_idx.numel()
        if num_pos < min_pos_per_anchor:
            continue
        if num_pos > max_pos_per_anchor:
            perm = torch.randperm(num_pos, device=device)
            pos_idx = pos_idx[perm[:max_pos_per_anchor]]
            num_pos = pos_idx.numel()

        mask_neg = (labels_t != lab_i)
        mask_neg[i] = False
        neg_idx = torch.nonzero(mask_neg).squeeze(-1)
        num_neg = neg_idx.numel()
        if num_neg == 0:
            continue

        all_idx = torch.cat([pos_idx, neg_idx], dim=0)
        logits = sim[i, all_idx]
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum()
        exp_pos = exp_logits[:num_pos]
        log_prob = torch.log(exp_pos / denom)
        loss_i = -log_prob.mean()

        losses.append(loss_i)
        total_anchors += 1
        total_pos_pairs += num_pos
        total_neg_pairs += num_neg

    loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    stats = dict(
        loss=float(loss.item()),
        total_anchors=total_anchors,
        total_pos_pairs=total_pos_pairs,
        total_neg_pairs=total_neg_pairs,
    )
    return loss, stats


def build_geomE_feature(geom4, deltaE, use_delta_E=True):
    """把 [Δgeom_4, (ΔE)] 拼接并标准化，返回 g_std，仅用于 PCA 可视化。"""
    geom4 = np.asarray(geom4)
    if use_delta_E and (deltaE is not None):
        deltaE = np.asarray(deltaE).reshape(-1, 1)
        g_all = np.concatenate([geom4, deltaE], axis=1)
    else:
        g_all = geom4
    mean = g_all.mean(axis=0, keepdims=True)
    std = g_all.std(axis=0, keepdims=True) + 1e-8
    g_std = (g_all - mean) / std
    return g_std


def geomE_weight_matrix(geom_batch, deltaE_batch, sigma=1.0, use_delta_E=True):
    """计算 batch 内 [Δgeom_4,(ΔE)] 的高斯核权重矩阵。"""
    if use_delta_E and (deltaE_batch is not None):
        g = torch.cat([geom_batch, deltaE_batch], dim=1)
    else:
        g = geom_batch
    sq_norm = (g ** 2).sum(dim=1, keepdim=True)
    dist2 = sq_norm + sq_norm.T - 2 * g @ g.T
    dist2 = torch.clamp(dist2, min=0.0)
    if sigma <= 0:
        raise ValueError("sigma 必须 > 0")
    w = torch.exp(-dist2 / (2 * sigma * sigma))
    B = g.size(0)
    w[torch.arange(B), torch.arange(B)] = 0.0
    return w


def supcon_loss_geom_continuous(
    z_batch,
    batch_indices,
    geom4,
    deltaE,
    label_mask,
    temperature=0.1,
    sigma=1.0,
    use_delta_E=True,
    pos_weight_thresh=0.3,
    min_effective_pos_per_anchor=4,
):
    """基于 geomE 高斯核的连续 SupCon 损失（与原脚本相同）。"""
    device = z_batch.device
    B = z_batch.shape[0]

    geom_all = np.array(geom4)
    geom_batch_np = geom_all[batch_indices]
    geom_batch = torch.from_numpy(geom_batch_np).float().to(device)
    if use_delta_E and (deltaE is not None):
        de_all = np.array(deltaE).reshape(-1, 1)
        de_batch_np = de_all[batch_indices]
        de_batch = torch.from_numpy(de_batch_np).float().to(device)
    else:
        de_batch = None

    w = geomE_weight_matrix(geom_batch, de_batch, sigma=sigma, use_delta_E=use_delta_E)
    if label_mask is not None:
        mask_t = torch.from_numpy(label_mask.astype(np.float32)).to(device)
        w = w * mask_t

    sim = torch.matmul(z_batch, z_batch.T) / temperature

    losses = []
    total_anchors = 0
    total_effective_pos = 0
    for i in range(B):
        w_i = w[i]
        mask_pos = (w_i >= pos_weight_thresh)
        pos_idx = torch.nonzero(mask_pos).squeeze(-1)
        num_pos = pos_idx.numel()
        if num_pos < min_effective_pos_per_anchor:
            continue
        if w_i.sum().item() <= 0:
            continue
        logits = sim[i]
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum()
        exp_pos = exp_logits[pos_idx]
        w_pos = w_i[pos_idx]
        Z_i = w_pos.sum()
        if Z_i.item() <= 0:
            continue
        log_prob = torch.log(exp_pos / denom)
        loss_i = - (w_pos * log_prob).sum() / Z_i
        losses.append(loss_i)
        total_anchors += 1
        total_effective_pos += int(num_pos)

    loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    stats = dict(
        loss=float(loss.item()),
        total_anchors=total_anchors,
        total_effective_pos=total_effective_pos,
    )
    return loss, stats


def oversample_indices_by_groups(groups, target_per_group, rng=None):
    """对每个 group 抽 target_per_group 个样本（不足则过采样），然后拼接。"""
    if rng is None:
        rng = np.random.default_rng()
    selected = []
    for g, idx in groups.items():
        n = len(idx)
        if n == 0:
            continue
        if n >= target_per_group:
            chosen = rng.choice(idx, size=target_per_group, replace=False)
        else:
            chosen = rng.choice(idx, size=target_per_group, replace=True)
        selected.append(chosen)
    if not selected:
        return np.array([], dtype=int)
    sel = np.concatenate(selected)
    rng.shuffle(sel)
    return sel


def select_tsne_indices(N_total, max_points=None):
    """简化版：默认使用全部样本；若 max_points < N，总体随机下采样。"""
    if max_points is None or max_points >= N_total:
        return np.arange(N_total)
    rng = np.random.default_rng()
    return rng.choice(np.arange(N_total), size=max_points, replace=False)


# ======================================================================
# 6. t-SNE / PCA / 保存函数 （与原脚本一致，此处略注释）
# ======================================================================

def run_tsne(Z, tsne_indices, perplexity=30.0):
    Z_sub = Z[tsne_indices]
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords_sub = tsne.fit_transform(Z_sub)
    return coords_sub


def scatter_by_material_and_reaction(ax, coords_sub, tsne_indices,
                                     substrates, reaction_types, title):
    subs = np.array(substrates)
    rts = np.array(reaction_types)
    subs_sub = subs[tsne_indices]
    rts_sub = rts[tsne_indices]

    mask_zeolite = (subs_sub == "zeolite")
    mask_alloy = (subs_sub == "alloy")
    mask_oxide = (subs_sub == "oxide")
    mask_MN4 = (subs_sub == "MN4-graphene")
    mask_MOF = (subs_sub == "MOF")

    mask_CH = (rts_sub == "C-H_activation")
    mask_CO = (rts_sub == "C-O_coupling")
    mask_direct = (rts_sub == "CH4_CH3OH_direct_oxidation")

    def plot_mask(mask, color, marker, label):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        sc = ax.scatter(coords_sub[idx, 0], coords_sub[idx, 1],
                        c=color, marker=marker, s=SCATTER_SIZE,
                        label=label, alpha=0.8)
        return sc

    sc_list = []
    sc_list.append(plot_mask(mask_zeolite & mask_CH, 'blue', 'o', "zeolite C-H_activation"))
    sc_list.append(plot_mask(mask_zeolite & mask_CO, 'blue', '*', "zeolite C-O_coupling"))
    sc_list.append(plot_mask(mask_zeolite & mask_direct, 'blue', 's', "zeolite direct"))

    sc_list.append(plot_mask(mask_alloy & mask_CH, 'red', 'o', "alloy C-H_activation"))
    sc_list.append(plot_mask(mask_alloy & mask_CO, 'red', '*', "alloy C-O_coupling"))
    sc_list.append(plot_mask(mask_alloy & mask_direct, 'red', 's', "alloy direct"))

    sc_list.append(plot_mask(mask_oxide & mask_CH, 'green', 'o', "oxide C-H_activation"))
    sc_list.append(plot_mask(mask_oxide & mask_CO, 'green', '*', "oxide C-O_coupling"))
    sc_list.append(plot_mask(mask_oxide & mask_direct, 'green', 's', "oxide direct"))

    sc_list.append(plot_mask(mask_MN4 & mask_CH, 'orange', 'o', "MN4-graphene C-H_activation"))
    sc_list.append(plot_mask(mask_MN4 & mask_CO, 'orange', '*', "MN4-graphene C-O_coupling"))
    sc_list.append(plot_mask(mask_MN4 & mask_direct, 'orange', 's', "MN4-graphene direct"))

    sc_list.append(plot_mask(mask_MOF & mask_CH, 'purple', 'o', "MOF C-H_activation"))
    sc_list.append(plot_mask(mask_MOF & mask_CO, 'purple', '*', "MOF C-O_coupling"))
    sc_list.append(plot_mask(mask_MOF & mask_direct, 'purple', 's', "MOF direct"))

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    return [s for s in sc_list if s is not None]


def scatter_with_continuous_color(ax, coords_sub, values_sub,
                                  title, cmap="coolwarm",
                                  vmin=None, vmax=None):
    sc = ax.scatter(coords_sub[:, 0], coords_sub[:, 1],
                    c=values_sub, s=SCATTER_SIZE,
                    cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    return sc


def compute_geomE_pca1d_label(geom4, deltaE, use_delta_E=True):
    g_std = build_geomE_feature(geom4, deltaE, use_delta_E=use_delta_E)
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(g_std).reshape(-1)
    explained_ratio = float(pca.explained_variance_ratio_[0])
    pc1_min, pc1_max = pc1.min(), pc1.max()
    pc1_norm = (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8)
    print(f"[PCA-1D] 解释方差比 (PC1): {explained_ratio:.4f}")
    return pc1_norm, explained_ratio


def save_encoder_both_formats(encoder, struct_dim, proj_dim, state_path, ts_path):
    torch.save(encoder.state_dict(), state_path)
    print(f"[Save] state_dict -> {state_path}")
    model_cpu = ReactionEncoder(struct_dim=struct_dim,
                                proj_dim=proj_dim,
                                mode="concat_diff")
    model_cpu.load_state_dict(encoder.state_dict())
    model_cpu.eval()
    model_cpu.to("cpu")
    scripted = torch.jit.script(model_cpu)
    scripted.save(ts_path)
    print(f"[Save] TorchScript -> {ts_path}")


def analyze_geom_sigma_default(geom4, deltaE, use_delta_E=True,
                               num_sample_points=8000,
                               num_pairs=200000,
                               quantile=10.0):
    geom4 = np.asarray(geom4)
    if use_delta_E and (deltaE is not None):
        deltaE = np.asarray(deltaE).reshape(-1, 1)
        g_all = np.concatenate([geom4, deltaE], axis=1)
    else:
        g_all = geom4

    N = g_all.shape[0]
    N_sub = min(num_sample_points, N)
    idx = np.random.choice(N, size=N_sub, replace=False)
    g_sub = g_all[idx]

    M = min(num_pairs, N_sub * (N_sub - 1) // 2)
    dists = []
    for _ in range(M):
        i, j = np.random.choice(N_sub, size=2, replace=False)
        d = np.linalg.norm(g_sub[i] - g_sub[j])
        dists.append(d)
    dists = np.array(dists)

    print("[geom4+ΔE 距离分析]")
    print("  样本数:", N, "子集:", N_sub, "pair 数:", M)
    print("  平均:", dists.mean(), "中位数:", np.median(dists))
    print("  5%:", np.percentile(dists, 5),
          " 10%:", np.percentile(dists, 10),
          " 20%:", np.percentile(dists, 20),
          "95%:", np.percentile(dists, 95))

    d_q = np.percentile(dists, quantile)
    print(f"  推荐 geom_sigma ≈ {quantile}% 分位数: {d_q}")
    return float(d_q)


# ======================================================================
# 9. 主训练流程（Stage2 损失修改的核心部分）
# ======================================================================

def train_two_stage_and_geom_only(
    IS_embeddings,
    FS_embeddings,
    substrates,
    reaction_types,
    geom4,
    deltaE,
    IS_names,
    FS_names,
    max_stage1_epochs=50,
    max_stage2_epochs=300,          # 改为 300
    batch_size=128,
    temperature=0.1,
    device="cpu",
    # Stage1 多级 SupCon 参数
    stage1_min_pos=8,
    stage1_max_pos=48,
    lambda_rxn_stage1=1.0,
    lambda_coarse7_stage1=0.5,
    samples_per_reaction3=3000,
    samples_per_coarse7=3000,
    # Stage2 参数（新的：multilevel + geomE）
    lambda_stage2_multilevel=1.0,   # 对 (lambda_rxn * L_rxn + lambda_mat * L_coarse7) 的整体权重
    lambda_stage2_geom=1.0,         # geomE 的权重
    target_per_coarse7_stage2=4000,
    use_delta_E=True,
    tsne_perplexity=30.0,
    geom_sigma=None,
    geom_pos_thresh=0.3,
    geom_min_effective_pos=4,
):
    """
    统一训练入口：
      - Stage1：多级 coarse 对比学习 (reaction3 + coarse7)
      - Stage2：多级 SupCon (reaction3 + coarse7) + geom4+ΔE Gaussian SupCon
      - Geom-only：仅 geom4+ΔE 连续 SupCon（单独 encoder）
    """
    os.makedirs("loss_plots/stage1", exist_ok=True)
    os.makedirs("loss_plots/stage2", exist_ok=True)
    os.makedirs("loss_plots/geom_only", exist_ok=True)
    os.makedirs("tsne_plots", exist_ok=True)
    os.makedirs("loss_data/stage1", exist_ok=True)
    os.makedirs("loss_data/stage2", exist_ok=True)
    os.makedirs("loss_data/geom_only", exist_ok=True)

    N, D = IS_embeddings.shape
    subs = np.array(substrates)
    rts = np.array(reaction_types)

    coarse7_labels, reaction3_labels, coarse7_groups, reaction3_groups = \
        build_coarse7_and_rxn_groups(subs, rts)

    if geom_sigma is None:
        geom_sigma = analyze_geom_sigma_default(
            geom4=geom4,
            deltaE=deltaE,
            use_delta_E=use_delta_E,
            quantile=10.0,
        )
        print(f"[Stage2 & geom-only] 使用自动推荐 geom_sigma = {geom_sigma:.4f}")

    encoder_two_stage = ReactionEncoder(struct_dim=D, proj_dim=128,
                                        mode="concat_diff").to(device)
    encoder_stage1 = ReactionEncoder(struct_dim=D, proj_dim=128,
                                     mode="concat_diff").to(device)
    encoder_stage1.load_state_dict(encoder_two_stage.state_dict())
    encoder_geom_only = ReactionEncoder(struct_dim=D, proj_dim=128,
                                        mode="concat_diff").to(device)
    encoder_geom_only.load_state_dict(encoder_two_stage.state_dict())

    encoder_before = ReactionEncoder(struct_dim=D, proj_dim=128,
                                     mode="concat_diff").to(device)
    encoder_before.load_state_dict(encoder_two_stage.state_dict())

    optimizer = torch.optim.Adam(encoder_two_stage.parameters(), lr=1e-4)
    optimizer_geom_only = torch.optim.Adam(encoder_geom_only.parameters(), lr=1e-4)
    rng = np.random.default_rng()

    stage1_epoch_losses = []
    stage2_epoch_losses = []
    geom_only_epoch_losses = []

    # ----------------------- Stage1 （与原脚本完全相同）-----------------------
    print("\n=========== Stage1: reaction3 + coarse7 SupCon ===========")
    for epoch in range(max_stage1_epochs):
        sel_rxn = oversample_indices_by_groups(
            reaction3_groups, target_per_group=samples_per_reaction3, rng=rng
        )
        sel_c7 = oversample_indices_by_groups(
            coarse7_groups, target_per_group=samples_per_coarse7, rng=rng
        )
        sel_idx = np.unique(np.concatenate([sel_rxn, sel_c7]))
        rng.shuffle(sel_idx)
        M = len(sel_idx)

        encoder_two_stage.train()
        batch_losses = []

        print(f"\n[Stage1 epoch {epoch+1}/{max_stage1_epochs}] M = {M}")
        num_batches = int(np.ceil(M / batch_size))
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, M)
            b_idx = sel_idx[start:end]

            h_IS = torch.from_numpy(IS_embeddings[b_idx]).float().to(device)
            h_FS = torch.from_numpy(FS_embeddings[b_idx]).float().to(device)
            z = encoder_two_stage(h_IS, h_FS)

            loss_rxn, stats_rxn = supcon_loss_discrete_labels(
                z, b_idx, reaction3_labels,
                label_func=lambda idx_global: reaction3_label_of_idx(idx_global, rts),
                temperature=temperature,
                min_pos_per_anchor=stage1_min_pos,
                max_pos_per_anchor=stage1_max_pos,
            )

            loss_c7, stats_c7 = supcon_loss_discrete_labels(
                z, b_idx, coarse7_labels,
                label_func=lambda idx_global: coarse7_label_of_idx(idx_global, subs, rts),
                temperature=temperature,
                min_pos_per_anchor=stage1_min_pos,
                max_pos_per_anchor=stage1_max_pos,
            )

            if stats_rxn["total_anchors"] == 0 and stats_c7["total_anchors"] == 0:
                continue

            loss = lambda_rxn_stage1 * loss_rxn + \
                   lambda_coarse7_stage1 * loss_c7

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        stage1_epoch_losses.append(mean_loss)
        print(f"[Stage1 epoch {epoch+1}/{max_stage1_epochs}] mean_loss={mean_loss:.4f}")

        dat_path = f"loss_data/stage1/stage1_epoch_{epoch+1:03d}_batch_loss.dat"
        with open(dat_path, "w") as f:
            f.write("# batch_index  loss(reaction3+coarse7)\n")
            for i, v in enumerate(batch_losses, start=1):
                f.write(f"{i}  {v:.8f}\n")

        plt.figure()
        plt.plot(np.arange(1, len(batch_losses) + 1), batch_losses,
                 marker='o', linewidth=1)
        plt.xlabel("Batch index", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Stage1 loss", fontsize=LABEL_FONTSIZE)
        plt.title(f"Stage1 Epoch {epoch+1} Batch Loss", fontsize=TITLE_FONTSIZE)
        plt.tight_layout()
        plt.savefig(f"loss_plots/stage1/stage1_epoch_{epoch+1:03d}.jpg", dpi=300)
        plt.close()

    with open("loss_data/stage1_epoch_mean_loss.dat", "w") as f:
        f.write("# epoch  mean_stage1_loss\n")
        for e, v in enumerate(stage1_epoch_losses, start=1):
            f.write(f"{e}  {v:.8f}\n")

    plt.figure()
    plt.plot(np.arange(1, len(stage1_epoch_losses) + 1),
             stage1_epoch_losses, marker='o', linewidth=1)
    plt.xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Mean Stage1 loss", fontsize=LABEL_FONTSIZE)
    plt.title("Stage1 epoch-level loss", fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig("loss_plots/stage1_overall.jpg", dpi=300)
    plt.close()

    encoder_stage1.load_state_dict(encoder_two_stage.state_dict())
    save_encoder_both_formats(
        encoder_stage1, struct_dim=D, proj_dim=128,
        state_path="reaction_encoder_stage1_multilevel.pth",
        ts_path="reaction_encoder_stage1_multilevel.pt",
    )

    # ----------------------- Stage2：新的多级 + geomE 联合损失 ----------------
    print("\n=========== Stage2: multilevel (rxn+coarse7) + geom4+ΔE ===========")
    for epoch in range(max_stage2_epochs):
        sel_idx = oversample_indices_by_groups(
            coarse7_groups, target_per_group=target_per_coarse7_stage2, rng=rng
        )
        rng.shuffle(sel_idx)
        M = len(sel_idx)

        encoder_two_stage.train()
        batch_losses = []

        print(f"\n[Stage2 epoch {epoch+1}/{max_stage2_epochs}] M = {M}")
        num_batches = int(np.ceil(M / batch_size))
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, M)
            b_idx = sel_idx[start:end]

            h_IS = torch.from_numpy(IS_embeddings[b_idx]).float().to(device)
            h_FS = torch.from_numpy(FS_embeddings[b_idx]).float().to(device)
            z = encoder_two_stage(h_IS, h_FS)

            # 1) 再次计算 reaction3 SupCon（保护 3 个反应大簇）
            loss_rxn, stats_rxn = supcon_loss_discrete_labels(
                z, b_idx, reaction3_labels,
                label_func=lambda idx_global: reaction3_label_of_idx(idx_global, rts),
                temperature=temperature,
                min_pos_per_anchor=stage1_min_pos,
                max_pos_per_anchor=stage1_max_pos,
            )

            # 2) 再次计算 coarse7 SupCon（保护 7 个 coarse 小簇）
            loss_c7, stats_c7 = supcon_loss_discrete_labels(
                z, b_idx, coarse7_labels,
                label_func=lambda idx_global: coarse7_label_of_idx(idx_global, subs, rts),
                temperature=temperature,
                min_pos_per_anchor=stage1_min_pos,
                max_pos_per_anchor=stage1_max_pos,
            )

            # 如果 multilevel 部分完全没有 anchor，则不计算该项
            multilevel_has_anchor = (stats_rxn["total_anchors"] > 0 or
                                     stats_c7["total_anchors"] > 0)

            # 3) 构造 geomE 同 coarse7 的 mask
            labels_batch = np.array(
                [coarse7_label_of_idx(int(i), subs, rts) for i in b_idx],
                dtype=int,
            )
            B = len(b_idx)
            mask_geom = np.zeros((B, B), dtype=np.int32)
            for i in range(B):
                for j in range(B):
                    if i == j:
                        continue
                    if labels_batch[i] >= 0 and labels_batch[i] == labels_batch[j]:
                        mask_geom[i, j] = 1

            # 4) 计算 geomE 连续 SupCon
            loss_geom, stats_geom = supcon_loss_geom_continuous(
                z, b_idx,
                geom4=geom4,
                deltaE=deltaE,
                label_mask=mask_geom,
                temperature=temperature,
                sigma=geom_sigma,
                use_delta_E=use_delta_E,
                pos_weight_thresh=geom_pos_thresh,
                min_effective_pos_per_anchor=geom_min_effective_pos,
            )

            if (not multilevel_has_anchor) and stats_geom["total_anchors"] == 0:
                continue

            # Stage2 总损失：
            #   L_stage2 = λ_stage2_multilevel * (λ_rxn_stage1 L_rxn
            #                                     + λ_coarse7_stage1 L_coarse7)
            #              + λ_stage2_geom * L_geomE
            multilevel_loss = 0.0
            if multilevel_has_anchor:
                multilevel_loss = (lambda_rxn_stage1 * loss_rxn +
                                   lambda_coarse7_stage1 * loss_c7)
            loss = lambda_stage2_multilevel * multilevel_loss + \
                   lambda_stage2_geom * loss_geom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        stage2_epoch_losses.append(mean_loss)
        print(f"[Stage2 epoch {epoch+1}/{max_stage2_epochs}] mean_loss={mean_loss:.4f}")

        dat_path = f"loss_data/stage2/stage2_epoch_{epoch+1:03d}_batch_loss.dat"
        with open(dat_path, "w") as f:
            f.write("# batch_index  loss(multilevel+geom4+ΔE)\n")
            for i, v in enumerate(batch_losses, start=1):
                f.write(f"{i}  {v:.8f}\n")

        plt.figure()
        plt.plot(np.arange(1, len(batch_losses) + 1), batch_losses,
                 marker='o', linewidth=1)
        plt.xlabel("Batch index", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Stage2 loss", fontsize=LABEL_FONTSIZE)
        plt.title(f"Stage2 Epoch {epoch+1} Batch Loss", fontsize=TITLE_FONTSIZE)
        plt.tight_layout()
        plt.savefig(f"loss_plots/stage2/stage2_epoch_{epoch+1:03d}.jpg", dpi=300)
        plt.close()

    with open("loss_data/stage2_epoch_mean_loss.dat", "w") as f:
        f.write("# epoch  mean_stage2_loss\n")
        for e, v in enumerate(stage2_epoch_losses, start=1):
            f.write(f"{e}  {v:.8f}\n")

    plt.figure()
    plt.plot(np.arange(1, len(stage2_epoch_losses) + 1),
             stage2_epoch_losses, marker='o', linewidth=1)
    plt.xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Mean Stage2 loss", fontsize=LABEL_FONTSIZE)
    plt.title("Stage2 epoch-level loss", fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig("loss_plots/stage2_overall.jpg", dpi=300)
    plt.close()

    save_encoder_both_formats(
        encoder_two_stage, struct_dim=D, proj_dim=128,
        state_path="reaction_encoder_two_stage_multilevel_geomE.pth",
        ts_path="reaction_encoder_two_stage_multilevel_geomE.pt",
    )

    # ---------------- Geom-only 分支：保持不变 ----------------
    print("\n=========== Geom-only: 仅 geom4+ΔE 连续 SupCon ===========")
    for epoch in range(max_stage2_epochs):
        sel_idx = np.arange(N)
        rng.shuffle(sel_idx)
        M = len(sel_idx)

        encoder_geom_only.train()
        batch_losses = []

        print(f"\n[Geom-only epoch {epoch+1}/{max_stage2_epochs}] M = {M}")
        num_batches = int(np.ceil(M / batch_size))
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, M)
            b_idx = sel_idx[start:end]

            h_IS = torch.from_numpy(IS_embeddings[b_idx]).float().to(device)
            h_FS = torch.from_numpy(FS_embeddings[b_idx]).float().to(device)
            z = encoder_geom_only(h_IS, h_FS)

            loss_geom, stats_geom = supcon_loss_geom_continuous(
                z, b_idx,
                geom4=geom4,
                deltaE=deltaE,
                label_mask=None,
                temperature=temperature,
                sigma=geom_sigma,
                use_delta_E=use_delta_E,
                pos_weight_thresh=geom_pos_thresh,
                min_effective_pos_per_anchor=geom_min_effective_pos,
            )
            if stats_geom["total_anchors"] == 0:
                continue

            optimizer_geom_only.zero_grad()
            loss_geom.backward()
            optimizer_geom_only.step()
            batch_losses.append(loss_geom.item())

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        geom_only_epoch_losses.append(mean_loss)
        print(f"[Geom-only epoch {epoch+1}/{max_stage2_epochs}] mean_loss={mean_loss:.4f}")

        dat_path = f"loss_data/geom_only/geom_only_epoch_{epoch+1:03d}_batch_loss.dat"
        with open(dat_path, "w") as f:
            f.write("# batch_index  loss(geom4+ΔE only)\n")
            for i, v in enumerate(batch_losses, start=1):
                f.write(f"{i}  {v:.8f}\n")

        plt.figure()
        plt.plot(np.arange(1, len(batch_losses) + 1), batch_losses,
                 marker='o', linewidth=1)
        plt.xlabel("Batch index", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Geom-only loss", fontsize=LABEL_FONTSIZE)
        plt.title(f"Geom-only Epoch {epoch+1} Batch Loss", fontsize=TITLE_FONTSIZE)
        plt.tight_layout()
        plt.savefig(f"loss_plots/geom_only/geom_only_epoch_{epoch+1:03d}.jpg", dpi=300)
        plt.close()

    with open("loss_data/geom_only_epoch_mean_loss.dat", "w") as f:
        f.write("# epoch  mean_geom_only_loss\n")
        for e, v in enumerate(geom_only_epoch_losses, start=1):
            f.write(f"{e}  {v:.8f}\n")

    plt.figure()
    plt.plot(np.arange(1, len(geom_only_epoch_losses) + 1),
             geom_only_epoch_losses, marker='o', linewidth=1)
    plt.xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Mean geom-only loss", fontsize=LABEL_FONTSIZE)
    plt.title("Geom-only epoch-level loss", fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig("loss_plots/geom_only_overall.jpg", dpi=300)
    plt.close()

    save_encoder_both_formats(
        encoder_geom_only, struct_dim=D, proj_dim=128,
        state_path="reaction_encoder_geomE_only.pth",
        ts_path="reaction_encoder_geomE_only.pt",
    )

    # ---------------- 计算四套 embedding + t-SNE 部分与原脚本相同 --------------
    Z_before = compute_reaction_embeddings(IS_embeddings, FS_embeddings,
                                           encoder_before, device=device)
    Z_stage1 = compute_reaction_embeddings(IS_embeddings, FS_embeddings,
                                           encoder_stage1, device=device)
    Z_two_stage = compute_reaction_embeddings(IS_embeddings, FS_embeddings,
                                              encoder_two_stage, device=device)
    Z_geom_only = compute_reaction_embeddings(IS_embeddings, FS_embeddings,
                                              encoder_geom_only, device=device)

    print("\n=========== t-SNE 可视化 (coarse + PCA-1D) ===========")
    tsne_indices = select_tsne_indices(N_total=N, max_points=None)
    coords_before = run_tsne(Z_before, tsne_indices, perplexity=tsne_perplexity)
    coords_stage1 = run_tsne(Z_stage1, tsne_indices, perplexity=tsne_perplexity)
    coords_two_stage = run_tsne(Z_two_stage, tsne_indices, perplexity=tsne_perplexity)
    coords_geom_only = run_tsne(Z_geom_only, tsne_indices, perplexity=tsne_perplexity)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    scatter_by_material_and_reaction(axes[0], coords_before, tsne_indices,
                                     substrates, reaction_types,
                                     "Before training")
    scatter_by_material_and_reaction(axes[1], coords_stage1, tsne_indices,
                                     substrates, reaction_types,
                                     "After Stage1 (multilevel)")
    scatter_by_material_and_reaction(axes[2], coords_two_stage, tsne_indices,
                                     substrates, reaction_types,
                                     "After Stage2 (multilevel+geomE)")
    scatter_by_material_and_reaction(axes[3], coords_geom_only, tsne_indices,
                                     substrates, reaction_types,
                                     "Geom-only (geom4+ΔE)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    plt.savefig("tsne_plots/tsne_four_embeddings_coarse_view.jpg", dpi=300)
    plt.close()

    pca1d_labels, pca_ratio = compute_geomE_pca1d_label(
        geom4, deltaE, use_delta_E=use_delta_E
    )
    p_sub = pca1d_labels[tsne_indices]
    vmin = np.nanpercentile(p_sub, 1)
    vmax = np.nanpercentile(p_sub, 99)

    fig = plt.figure(figsize=(24, 5))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.03])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    cax = fig.add_subplot(gs[4])

    sc0 = scatter_with_continuous_color(
        ax0, coords_before, p_sub,
        title=f"Before (PCA1D, var={pca_ratio:.3f})",
        cmap="coolwarm", vmin=vmin, vmax=vmax,
    )
    scatter_with_continuous_color(
        ax1, coords_stage1, p_sub,
        title="Stage1 (PCA1D)",
        cmap="coolwarm", vmin=vmin, vmax=vmax,
    )
    scatter_with_continuous_color(
        ax2, coords_two_stage, p_sub,
        title="Stage2 (PCA1D)",
        cmap="coolwarm", vmin=vmin, vmax=vmax,
    )
    scatter_with_continuous_color(
        ax3, coords_geom_only, p_sub,
        title="Geom-only (PCA1D)",
        cmap="coolwarm", vmin=vmin, vmax=vmax,
    )
    cb = fig.colorbar(sc0, cax=cax)
    cb.set_label("PCA-1D label (normalized)", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()
    plt.savefig("tsne_plots/tsne_four_embeddings_pca1d_color.jpg", dpi=300)
    plt.close()

    return (encoder_two_stage, encoder_stage1, encoder_geom_only,
            Z_before, Z_stage1, Z_two_stage, Z_geom_only)


# ======================================================================
# 10. main
# ======================================================================

if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    set_seed(42)
    device = "cpu"  # 有 GPU 可改成 "cuda"

    (IS_emb, FS_emb,
     subs, rxn_types,
     IS_names, FS_names) = load_struct_embeddings_and_reactions("embeddings")

    geom4, deltaE = load_geom4_and_deltaE("embeddings")

    print(f"IS_embeddings 规模: N={IS_emb.shape[0]}, D={IS_emb.shape[1]}")

    (enc_two_stage,
     enc_stage1,
     enc_geom_only,
     Z_before,
     Z_stage1,
     Z_two_stage,
     Z_geom_only) = train_two_stage_and_geom_only(
        IS_embeddings=IS_emb,
        FS_embeddings=FS_emb,
        substrates=subs,
        reaction_types=rxn_types,
        geom4=geom4,
        deltaE=deltaE,
        IS_names=IS_names,
        FS_names=FS_names,
        max_stage1_epochs=100,
        max_stage2_epochs=500,          # Stage2 默认改为 300 epoch
        batch_size=128,
        temperature=0.1,
        device=device,
        stage1_min_pos=8,
        stage1_max_pos=48,
        lambda_rxn_stage1=1.0,
        lambda_coarse7_stage1=0.5,
        samples_per_reaction3=4000,
        samples_per_coarse7=4000,
        lambda_stage2_multilevel=0.2,   # 你可以后续调小/调大
        lambda_stage2_geom=1.0,
        target_per_coarse7_stage2=4000,
        use_delta_E=True,
        tsne_perplexity=30.0,
        geom_sigma=None,
        geom_pos_thresh=0.3,
        geom_min_effective_pos=4,
    )

    np.save("reaction_embeddings_before_training_128.npy", Z_before)
    np.save("reaction_embeddings_stage1_multilevel_128.npy", Z_stage1)
    np.save("reaction_embeddings_two_stage_multilevel_geomE_128.npy", Z_two_stage)
    np.save("reaction_embeddings_geomE_only_128.npy", Z_geom_only)

    print("\n四套 reaction embedding 已保存：")
    print("  - reaction_embeddings_before_training_128.npy")
    print("  - reaction_embeddings_stage1_multilevel_128.npy")
    print("  - reaction_embeddings_two_stage_multilevel_geomE_128.npy")
    print("  - reaction_embeddings_geomE_only_128.npy")
    print("对应 encoder：")
    print("  - reaction_encoder_stage1_multilevel.pth / .pt")
    print("  - reaction_encoder_two_stage_multilevel_geomE.pth / .pt")
    print("  - reaction_encoder_geomE_only.pth / .pt")
    print("以及 loss_plots/, loss_data/, tsne_plots/ 等辅助文件，可用于分析训练过程。")
