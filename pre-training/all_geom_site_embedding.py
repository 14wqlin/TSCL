# -*- coding: utf-8 -*-
"""
all_geom_site_embedding.py

功能：将几何计算 (Part1) + 活性位截取 + CGCNN embedding (Part2) 整合，
      针对下游势垒数据集 879-9762-0330-reaction-list.csv，
      同时支持 5 种材料：
        - zeolite
        - alloy
        - oxide
        - MN4-graphene
        - MOF

本版本在原脚本基础上做了重要修订（对应你的 7 条需求 + 新增 d_CO_star 特例）：

1）所有几何向量 (9 维) 的距离相关量，全部改为“考虑周期性边界条件(PBC)的最小
   镜像距离”，使用 ASE 的 atoms.get_distances(mic=True) 实现。

2）d_CM_min 的计算更新为使用 PBC 距离，解决 m‑dobdc / BBTA 等 MOF 中 C‑M 距离
   被错误算到 ~11 Å / ~6 Å 的问题。

3）CN_CH、max_d_CH 更新为使用 PBC 距离；并且对沸石中 CH4 中间体增加特判：
      - 对于沸石 substrate:
          sub_idx = set(get_substrate_indices_zeolite(atoms))
          all_idx = set(range(n_atoms))
          active_idx = sorted(list(all_idx - sub_idx))
          c_idx = [i for i in active_idx if symbols[i] == "C"]
          h_idx = [i for i in active_idx if symbols[i] == "H"]
        如果 h_idx 数量为 5 且 CN_CH == 4，则强制 max_d_CH = 1.10 Å，
        避免把活性位点上的 H 也算进去导致 CH4 的 C-H 最大键长被拉大。

4）d_CO_star 更新为使用 PBC 距离；并新增一个针对沸石 CH3OH 的特例：
      - 对于沸石 substrate 的几何计算（使用 FS 结构）：
          sub_idx = set(get_substrate_indices_zeolite(atoms_FS))
          all_idx = set(range(n_atoms_FS))
          active_idx = sorted(list(all_idx - sub_idx))
          c_idx = [i for i in active_idx if symbols[i] == "C"]
          h_idx = [i for i in active_idx if symbols[i] == "H"]
        如果 active_idx 中 H 的数量为 5 且 FS_adsorbate == "CH3OH"，
        则强制将 FS 的 d_CO_star 设为 1.48 Å。

5）CN_M 的判断阈值改为 d_OM < 3.0 Å（原为 2.6 Å）。

6）Zeolite 中 MOO 型活性 O 的索引：
      if nO == 2 and nM == 1:
          if 96 < n_atoms:
              return 96
   这里将 96 改成 97（0-based），即 1-based 的 98 号原子。

7）min_d_OH 的定义修改：
   - 原脚本只在 d_OH < 1.3 Å 的那一组中取最小值，导致很多结构为 0。
   - 现改为对“所有中间体 H 原子”计算活性 O 到 H 的距离（仍按 PBC），
     然后取最小值；如果结构中没有 H 或未找到活性 O，则设为 0。
   - CN_OH 仍旧定义为 d_OH < 1.3 Å 的 H 的个数。

8）所有原本输出在 reg/ 目录下的文件（geom_*.npy、embedding、active_site、
   直方图等），现统一输出在 embeddings/ 目录下。

其余功能（打印、保存、作图等）均保留，并增加了更详细的注释。
"""

import os
import random
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
import matplotlib.pyplot as plt

import torch
from ocpmodels.models import CGCNN
from ocpmodels.preprocessing import AtomsToGraphs


# ======================================================================
# 0. 通用工具
# ======================================================================

def set_seed(seed=42):
    """固定随机种子，保证实验可复现。"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pbc_distance(atoms, i, j):
    """
    计算原子 i, j 在 PBC 下的最小镜像距离。
    使用 ASE 内置的 atoms.get_distances(i, j, mic=True)：
      - mic=True: minimum image convention
      - 能自动处理任意晶格（含非正交）和 3D PBC
    """
    d = atoms.get_distances(i, j, mic=True)
    return float(np.array(d).ravel()[0])


# ======================================================================
# 1. CSV 读取
# ======================================================================

def _split_int_list(s):
    """将 '7,8,9,10'、'55'、'216.0' 或 216.0 解析为 int 列表。NaN/空返回 None。"""
    if isinstance(s, float):
        if np.isnan(s):
            return None
        return [int(round(s))]

    if s is None:
        return None

    s = str(s).strip()
    if len(s) == 0:
        return None

    parts = s.split(",")
    out = []
    for p in parts:
        p = p.strip()
        if p == "":
            continue
        try:
            val = int(float(p))
        except ValueError:
            raise ValueError(f"_split_int_list 解析失败: '{p}' 来自原始字符串 '{s}'")
        out.append(val)
    return out if len(out) > 0 else None


def load_reaction_csv_all(csv_path):
    """
    读取 879-9762-0330-reaction-list.csv（或同格式的 CSV）。

    返回：
      - rows: list[dict]，每行一个 dict，供几何和局部截取使用；
      - df  : 原始 DataFrame，方便另存或调试。
    """
    df = pd.read_csv(csv_path)

    required_basic = [
        "reaction_id", "reaction_type", "substrate",
        "IS_adsorbate", "FS_adsorbate",
        "IS_name", "FS_name",
        "IS_energy_mlp(eV)", "FS_energy_mlp(eV)",
        "IS_path", "FS_path",
    ]
    for c in required_basic:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少列: {c}")

    optional_energy = [
        "deltaE_mlp(eV)", "deltaE_DFT(eV)", "Ea_DFT(eV)",
    ]
    for c in optional_energy:
        if c not in df.columns:
            print(f"[警告] CSV 未找到列 {c}，将用 NaN 填充。")
            df[c] = np.nan

    rows = []
    for _, r in df.iterrows():
        row = dict(
            reaction_id=int(r["reaction_id"]),
            reaction_type=str(r["reaction_type"]).strip(),
            substrate=str(r["substrate"]).strip(),
            IS_adsorbate=str(r["IS_adsorbate"]).strip(),
            FS_adsorbate=str(r["FS_adsorbate"]).strip(),
            IS_name=str(r["IS_name"]).strip(),
            FS_name=str(r["FS_name"]).strip(),
            IS_energy_mlp=float(r["IS_energy_mlp(eV)"]),
            FS_energy_mlp=float(r["FS_energy_mlp(eV)"]),
            deltaE_mlp=float(r["deltaE_mlp(eV)"])
            if not pd.isna(r["deltaE_mlp(eV)"]) else np.nan,
            deltaE_DFT=float(r["deltaE_DFT(eV)"])
            if not pd.isna(r["deltaE_DFT(eV)"]) else np.nan,
            Ea_DFT=float(r["Ea_DFT(eV)"])
            if not pd.isna(r["Ea_DFT(eV)"]) else np.nan,
            IS_path=str(r["IS_path"]).strip(),
            FS_path=str(r["FS_path"]).strip(),
        )

        # index 列
        row["O_index_1based"] = int(r["O_index_1based"]) if (
            "O_index_1based" in df.columns and not pd.isna(r["O_index_1based"])
        ) else None

        row["C_index_1based"] = int(r["C_index_1based"]) if (
            "C_index_1based" in df.columns and not pd.isna(r["C_index_1based"])
        ) else None

        row["H_index_1based"] = _split_int_list(r["H_index_1based"]) if (
            "H_index_1based" in df.columns
        ) else None

        row["M_index_1based"] = _split_int_list(r["M_index_1based"]) if (
            "M_index_1based" in df.columns
        ) else None

        row["sub_index_1based"] = _split_int_list(r["sub_index_1based"]) if (
            "sub_index_1based" in df.columns
        ) else None

        rows.append(row)
    return rows, df


# ======================================================================
# 2. 几何计算
# ======================================================================

def get_substrate_indices_zeolite(structure):
    """
    定义“沸石骨架”的原子索引，用于从全结构中区分骨架与活性位。

    规则：
      - 前 96 个原子；
      - 所有 Si / Al；
    """
    substrate_indices = set(range(96))
    si_idx = [i for i, a in enumerate(structure) if a.symbol == "Si"]
    al_idx = [i for i, a in enumerate(structure) if a.symbol == "Al"]
    substrate_indices.update(si_idx)
    substrate_indices.update(al_idx)
    return sorted(list(substrate_indices))


def get_active_O_index_zeolite_by_geometry(atoms):
    """
    根据骨架/非骨架划分 + 元素统计推断沸石中的活性 O index (0-based)。

    注意：这里按照你的要求，MOO 中活性 O 改为 0-based 的 97（1-based 98）。
    """
    n_atoms = len(atoms)
    if n_atoms <= 96:
        return None

    sub_idx = set(get_substrate_indices_zeolite(atoms))
    all_idx = set(range(n_atoms))
    active_idx = sorted(list(all_idx - sub_idx))
    if len(active_idx) == 0:
        return None

    symbols = atoms.get_chemical_symbols()
    o_idx = [i for i in active_idx if symbols[i] == "O"]
    h_idx = [i for i in active_idx if symbols[i] == "H"]
    m_idx = [
        i for i in active_idx
        if symbols[i] not in ["C", "H", "O", "Si", "Al"]
    ]

    nO = len(o_idx)
    nM = len(m_idx)

    # 只有一个 O：直接是活性 O
    if nO == 1:
        return o_idx[0]

    # MOO: 两个 O + 一个 M，对应 1-based 98 号原子 => 0-based 97
    if nO == 2 and nM == 1:
        if 97 < n_atoms:
            return 97
        else:
            return None

    # MOOH: 两个 O + 两个 M，对应 1-based 98 号原子 => 0-based 97
    if nO == 2 and nM == 2:
        if 97 < n_atoms:
            return 97
        else:
            return None

    return None


def compute_geom_vector_full_structure(
    atoms,
    substrate,
    adsorbate_type,
    active_O_index=None,
    main_C_index=None,
    override_idx_H=None,
    override_idx_M=None,
):
    """
    在完整结构上围绕“反应中心”计算 9 维几何向量 geom。

    9 维定义：
      0: d_CM_min   (C 到所有金属的最小距离，PBC)
      1: CN_CH      (C-H 配位数，d_CH < 1.3 Å，PBC)
      2: max_d_CH   (C 到所有 H 的最大距离，PBC；
                     对 zeolite + CH4 特例：若 H 数=5 且 CN_CH=4，则强制 1.10 Å)
      3: CN_CO      (C-O 配位数，d_CO < 1.6 Å，PBC)
      4: d_CO_star  (C 到主 O* 的距离，PBC)
      5: d_OM_min   (主 O* 到所有金属的最小距离，PBC)
      6: CN_M       (主 O* 到金属配位数，d_OM < 3.0 Å，PBC)
      7: CN_OH      (O-H 配位数，d_OH < 1.3 Å，PBC)
      8: min_d_OH   (主 O* 到中间体 H 的最近距离，PBC；不再限制 d_OH<1.3)
    """
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    # 全局 C/O/H 索引
    idx_C_all = [i for i, s in enumerate(symbols) if s == "C"]
    idx_O_all = [i for i, s in enumerate(symbols) if s == "O"]

    if override_idx_H is not None:
        idx_H_all = list(override_idx_H)
    else:
        idx_H_all = [i for i, s in enumerate(symbols) if s == "H"]

    # 金属索引
    if override_idx_M is not None:
        idx_M_all = list(override_idx_M)
    else:
        if substrate == "zeolite":
            idx_M_all = [i for i, s in enumerate(symbols)
                         if s not in ["C", "H", "O", "Si", "Al"]]
        elif substrate == "MN4-graphene":
            idx_M_all = [i for i, s in enumerate(symbols)
                         if s not in ["C", "H", "N", "O"]]
        elif substrate == "oxide":
            idx_M_all = [i for i, s in enumerate(symbols)
                         if s not in ["C", "H", "O"]]
        elif substrate == "MOF":
            idx_M_all = [i for i, s in enumerate(symbols)
                         if s not in ["C", "H", "O", "N"]]
        else:  # alloy 等
            idx_M_all = [i for i, s in enumerate(symbols)
                         if s not in ["C", "H", "O"]]

    def dist(i, j):
        return pbc_distance(atoms, i, j)

    feats = []

    # ---------- A. 确定主 C index ----------
    if main_C_index is not None:
        c_idx = main_C_index
    else:
        c_idx = idx_C_all[0] if len(idx_C_all) > 0 else None

    # ---------- 0. d_CM_min ----------
    if c_idx is not None and len(idx_M_all) > 0:
        d_CM = np.array([dist(c_idx, m) for m in idx_M_all])
        d_CM_min = float(d_CM.min())
    else:
        d_CM_min = 0.0
    feats.append(d_CM_min)

    # ---------- 1. CN_CH，2. max_d_CH ----------
    CN_CH = 0
    max_d_CH = 0.0
    if c_idx is not None and len(idx_H_all) > 0:
        d_CH_all = np.array([dist(c_idx, h) for h in idx_H_all])
        max_d_CH = float(d_CH_all.max())
        d_CH_bonds = d_CH_all[d_CH_all < 1.3]
        CN_CH = int(len(d_CH_bonds))

        # 沸石中 CH4 特例：活性位/中间体中的 H 为 5，且 C 周围 H 配位数为 4
        if substrate == "zeolite":
            sub_idx = set(get_substrate_indices_zeolite(atoms))
            all_idx = set(range(n_atoms))
            active_idx = sorted(list(all_idx - sub_idx))
            c_active = [i for i in active_idx if symbols[i] == "C"]
            h_active = [i for i in active_idx if symbols[i] == "H"]
            if len(c_active) == 1 and len(h_active) == 5 and CN_CH == 4:
                # CH4 上应有 4 个 H，另 1 个 H 是活性位上的 H，不应拉高 max_d_CH
                max_d_CH = 1.10  # 强制设为 1.10 Å
    feats.append(float(CN_CH))
    feats.append(max_d_CH)

    # ---------- B. 确定主 O* index ----------
    if active_O_index is not None:
        o_ads = int(active_O_index)
    else:
        o_ads = None
        if c_idx is not None and len(idx_O_all) > 0:
            best_o, best_d = None, 1e9
            for o in idx_O_all:
                d_co = dist(c_idx, o)
                if d_co < best_d:
                    best_o, best_d = o, d_co
            o_ads = best_o

    # ---------- 3. CN_CO ----------
    CN_CO = 0
    if c_idx is not None and len(idx_O_all) > 0:
        for o in idx_O_all:
            if dist(c_idx, o) < 1.6:
                CN_CO += 1

    # ---------- 4. d_CO_star ----------
    d_CO_star = 0.0
    if c_idx is not None and o_ads is not None:
        d_CO_star = dist(c_idx, o_ads)

    feats.append(float(CN_CO))
    feats.append(float(d_CO_star))

    # ---------- 5. d_OM_min，6. CN_M ----------
    d_OM_min = 0.0
    CN_M = 0
    if o_ads is not None and len(idx_M_all) > 0:
        d_OM = np.array([dist(o_ads, m) for m in idx_M_all])
        d_OM_min = float(d_OM.min())
        # 阈值从 2.6 Å 改为 3.0 Å
        CN_M = int(np.sum(d_OM < 3.0))
    feats.append(d_OM_min)
    feats.append(float(CN_M))

    # ---------- 7. CN_OH，8. min_d_OH ----------
    CN_OH = 0
    min_d_OH = 0.0
    if o_ads is not None and len(idx_H_all) > 0:
        d_all = np.array([dist(o_ads, h) for h in idx_H_all])
        # CN_OH: d_OH < 1.3 Å 的个数
        CN_OH = int(np.sum(d_all < 1.3))
        # min_d_OH: 所有 H 中的最小 O-H 距离（不再限制 <1.3）
        min_d_OH = float(d_all.min())
    feats.append(float(CN_OH))
    feats.append(min_d_OH)

    return np.array(feats, dtype=np.float32)


# ======================================================================
# 3. 局部截取 + CGCNN embedding
# ======================================================================

def extract_catalytic_atoms_zeolite(vasp_file_path):
    """
    沿用原 part2_extract_and_embeddings_Ea_reg 的沸石局部截取规则：
      - 非骨架原子 + 与骨架 Al/O/Si 邻近的一圈原子。
    """
    full = ase.io.read(vasp_file_path, format="vasp")
    all_idx = set(range(len(full)))
    sub_idx = set(get_substrate_indices_zeolite(full))
    active_idx = sorted(list(all_idx - sub_idx))

    related = set()
    al_idx = [i for i, a in enumerate(full) if a.symbol == "Al"]
    related.update(al_idx)

    cutoff_al_o = 2.0
    i_idx, j_idx, d = neighbor_list("ijd", full, cutoff_al_o)
    o_near_al = set()
    for i_, j_, dist_ in zip(i_idx, j_idx, d):
        if full[i_].symbol == "Al" and full[j_].symbol == "O":
            o_near_al.add(j_)
        elif full[j_].symbol == "Al" and full[i_].symbol == "O":
            o_near_al.add(i_)
    related.update(o_near_al)

    cutoff_o_si = 2.0
    i_idx, j_idx, d = neighbor_list("ijd", full, cutoff_o_si)
    si_near_o = set()
    for i_, j_, dist_ in zip(i_idx, j_idx, d):
        if i_ in o_near_al and full[j_].symbol == "Si":
            si_near_o.add(j_)
        elif j_ in o_near_al and full[i_].symbol == "Si":
            si_near_o.add(i_)
    related.update(si_near_o)

    selected_set = set(active_idx) | related
    selected = sorted(list(selected_set))
    sub_atoms = full[selected]
    return sub_atoms, selected


def get_active_site_indices_binary_alloy(structure):
    """合金局部截取规则：C/H/O + 与 O 相邻的金属。"""
    active = set()
    for i, a in enumerate(structure):
        if a.symbol in ["C", "H", "O"]:
            active.add(i)
    oxy_idx = [i for i, a in enumerate(structure) if a.symbol == "O"]
    metal_idx = [i for i, a in enumerate(structure)
                 if a.symbol not in ["C", "H", "O"]]

    cutoff = 6.0
    i_idx, j_idx, d = neighbor_list("ijd", structure, cutoff)
    metal_near = set()
    for oi in oxy_idx:
        neigh = j_idx[i_idx == oi]
        for n in neigh:
            if n in metal_idx:
                metal_near.add(int(n))
    active.update(sorted(list(metal_near)))
    return sorted(list(active))


def extract_catalytic_atoms_binary_alloy(vasp_file_path):
    """合金局部截取。"""
    full = ase.io.read(vasp_file_path, format="vasp")
    active_idx = get_active_site_indices_binary_alloy(full)
    sub_atoms = full[active_idx]
    return sub_atoms, active_idx


def extract_catalytic_atoms_oxide(vasp_file_path, O_index_1based, cutoff_sub=6.0):
    """
    氧化物局部截取：
      - 活性 O* index 由 O_index_1based 给出；
      - 中间体原子：所有 C、所有 H、活性 O*；
      - 衬底原子：所有非 C/H 原子中，距离 O* < cutoff_sub 的原子；
      - 最终局部结构 = 中间体原子 ∪ 衬底近邻原子。
    """
    full = ase.io.read(vasp_file_path, format="vasp")
    symbols = full.get_chemical_symbols()
    n = len(full)

    if O_index_1based is None or O_index_1based < 1 or O_index_1based > n:
        raise ValueError(
            f"oxide 结构 {vasp_file_path} 未给出合法的 O_index_1based: {O_index_1based}"
        )
    o_idx = O_index_1based - 1

    idx_C = [i for i, s in enumerate(symbols) if s == "C"]
    idx_H = [i for i, s in enumerate(symbols) if s == "H"]
    idx_sub_all = [i for i, s in enumerate(symbols) if s not in ["C", "H"]]

    def dist(i, j):
        return pbc_distance(full, i, j)

    substrate_near = []
    for i in idx_sub_all:
        if dist(o_idx, i) < cutoff_sub:
            substrate_near.append(i)

    active_set = set(idx_C) | set(idx_H) | {o_idx} | set(substrate_near)
    selected = sorted(list(active_set))
    sub_atoms = full[selected]
    return sub_atoms, selected


def extract_catalytic_atoms_MN4_graphene(vasp_file_path, C_index_1based, cutoff_C=6.0):
    """
    MN4-graphene 局部截取：
      - CH4 中心 C index 由 C_index_1based 给定；
      - 中间体原子：CH4 中心 C + 所有 H + 所有 O + 所有 N + 所有金属 M；
      - 石墨烯 C 衬底：除 CH4 中心 C 外，距离金属 M < cutoff_C 的 C；
      - 最终局部结构 = 中间体原子 ∪ 石墨烯 C。
    """
    full = ase.io.read(vasp_file_path, format="vasp")
    symbols = full.get_chemical_symbols()
    n = len(full)

    if C_index_1based is None or C_index_1based < 1 or C_index_1based > n:
        raise ValueError(
            f"MN4-graphene 结构 {vasp_file_path} 未给出合法的 C_index_1based: {C_index_1based}"
        )
    c_ch4 = C_index_1based - 1

    idx_O = [i for i, s in enumerate(symbols) if s == "O"]
    idx_H = [i for i, s in enumerate(symbols) if s == "H"]
    idx_N = [i for i, s in enumerate(symbols) if s == "N"]
    idx_M = [i for i, s in enumerate(symbols)
             if s not in ["C", "H", "N", "O"]]
    idx_C_all = [i for i, s in enumerate(symbols) if s == "C"]

    if len(idx_O) != 1:
        print(f"[警告] MN4-graphene {vasp_file_path} 中 O 原子数为 {len(idx_O)}，期望 1。")
    if len(idx_M) != 1:
        print(f"[警告] MN4-graphene {vasp_file_path} 中金属原子数为 {len(idx_M)}，期望 1。")

    m_idx = idx_M[0] if len(idx_M) > 0 else None

    def dist(i, j):
        return pbc_distance(full, i, j)

    graphene_C = [i for i in idx_C_all if i != c_ch4]

    substrate_C = []
    if m_idx is not None:
        for ci in graphene_C:
            if dist(m_idx, ci) < cutoff_C:
                substrate_C.append(ci)

    active_set = {c_ch4} | set(idx_H) | set(idx_O) | set(idx_N) | set(idx_M) | set(substrate_C)
    selected = sorted(list(active_set))
    sub_atoms = full[selected]
    return sub_atoms, selected


def extract_catalytic_atoms_MOF(
    vasp_file_path,
    O_index_1based,
    C_index_1based,
    H_index_1based_list,
    M_index_1based_list,
    sub_index_1based_list,
):
    """
    MOF 局部截取：
      - 所有 index 均由 CSV 指定（1-based），函数负责转为 0-based 并合并去重。
    """
    full = ase.io.read(vasp_file_path, format="vasp")
    n = len(full)

    idx_set = set()

    if O_index_1based is not None:
        if not (1 <= O_index_1based <= n):
            raise ValueError(f"MOF {vasp_file_path} O_index_1based 超出范围: {O_index_1based}")
        idx_set.add(O_index_1based - 1)

    if C_index_1based is not None:
        if not (1 <= C_index_1based <= n):
            raise ValueError(f"MOF {vasp_file_path} C_index_1based 超出范围: {C_index_1based}")
        idx_set.add(C_index_1based - 1)

    if H_index_1based_list is not None:
        for h in H_index_1based_list:
            if not (1 <= h <= n):
                raise ValueError(f"MOF {vasp_file_path} H_index_1based 超出范围: {h}")
            idx_set.add(h - 1)

    if M_index_1based_list is not None:
        for m in M_index_1based_list:
            if not (1 <= m <= n):
                raise ValueError(f"MOF {vasp_file_path} M_index_1based 超出范围: {m}")
            idx_set.add(m - 1)

    if sub_index_1based_list is not None:
        for s in sub_index_1based_list:
            if not (1 <= s <= n):
                raise ValueError(f"MOF {vasp_file_path} sub_index_1based 超出范围: {s}")
            idx_set.add(s - 1)

    if len(idx_set) == 0:
        raise ValueError(f"MOF {vasp_file_path} 未提供任何局部原子 index。")

    selected = sorted(list(idx_set))
    sub_atoms = full[selected]
    return sub_atoms, selected


def vasp_to_feature_vector_selected(atoms, model):
    """
    使用 CGCNN 对局部结构 atoms 计算图级 embedding。
    """
    a2g = AtomsToGraphs(
        max_neigh=20,
        radius=model.cutoff,
        r_edges=True,
        r_fixed=True,
        r_pbc=True,
        r_energy=False,
        r_forces=False,
        r_distances=False,
    )
    data = a2g.convert(atoms)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    device = next(model.parameters()).device
    data = data.to(device)

    data.x = model.embedding[data.atomic_numbers.long() - 1]
    dists = torch.norm(
        data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]], dim=1
    )
    data.edge_attr = model.distance_expansion(dists)

    with torch.no_grad():
        h = model._convolve(data)
    return h.cpu().numpy().reshape(-1)


class StructureEncoder:
    """
    封装 CGCNN，对每个 VASP 文件 (IS/FS) 输出局部 embedding。
    """

    def __init__(self, cgcnn_model, device="cpu"):
        self.model = cgcnn_model.to(device)
        self.model.eval()

    def encode_local(self, vasp_path, reaction_row):
        sub = reaction_row["substrate"]

        if sub == "zeolite":
            atoms_local, _ = extract_catalytic_atoms_zeolite(vasp_path)

        elif sub == "alloy":
            atoms_local, _ = extract_catalytic_atoms_binary_alloy(vasp_path)

        elif sub == "oxide":
            O1b = reaction_row.get("O_index_1based", None)
            atoms_local, _ = extract_catalytic_atoms_oxide(vasp_path, O1b)

        elif sub == "MN4-graphene":
            C1b = reaction_row.get("C_index_1based", None)
            atoms_local, _ = extract_catalytic_atoms_MN4_graphene(vasp_path, C1b)

        elif sub == "MOF":
            O1b = reaction_row.get("O_index_1based", None)
            C1b = reaction_row.get("C_index_1based", None)
            H1b_list = reaction_row.get("H_index_1based", None)
            M1b_list = reaction_row.get("M_index_1based", None)
            sub1b_list = reaction_row.get("sub_index_1based", None)
            atoms_local, _ = extract_catalytic_atoms_MOF(
                vasp_path,
                O1b,
                C1b,
                H1b_list,
                M1b_list,
                sub1b_list,
            )
        else:
            raise ValueError(f"未知 substrate: {sub}")

        h = vasp_to_feature_vector_selected(atoms_local, self.model)
        return h, atoms_local


# ======================================================================
# 4. 主流程：几何 + 局部截取 + embedding
# ======================================================================

def prepare_Ea_reg_all(
    csv_path="879-9762-0330-reaction-list.csv",
    out_dir="embeddings",      # 输出目录
    device="cpu",
):
    os.makedirs(out_dir, exist_ok=True)

    # --- 读 CSV ---
    reactions, df_full = load_reaction_csv_all(csv_path)
    N = len(reactions)
    print(f"总反应数: {N}")

    df_full.to_csv(os.path.join(out_dir, "full_table.csv"), index=False)

    # --- 初始化 CGCNN ---
    cgcnn_model = CGCNN(
        num_atoms=170,
        bond_feat_dim=50,
        num_targets=1,
        use_pbc=True,
        regress_forces=False,
        atom_embedding_size=64,
        num_graph_conv_layers=6,
        fc_feat_size=128,
        num_fc_layers=4,
        otf_graph=False,
        cutoff=4.0,
        num_gaussians=50,
    ).to(device)
    cgcnn_model.eval()
    encoder = StructureEncoder(cgcnn_model, device=device)

    # --- active_site 输出目录 ---
    active_site_dir = os.path.join(out_dir, "active_site")
    os.makedirs(active_site_dir, exist_ok=True)
    for sub in ["zeolite", "alloy", "oxide", "MN4-graphene", "MOF"]:
        os.makedirs(os.path.join(active_site_dir, sub), exist_ok=True)

    # --- 用第一条反应确定 embedding 维度 ---
    tmp_h, _ = encoder.encode_local(reactions[0]["IS_path"], reactions[0])
    D = tmp_h.shape[0]
    print(f"CGCNN 局部 embedding 维度: {D}")

    IS_emb = np.zeros((N, D), np.float32)
    FS_emb = np.zeros((N, D), np.float32)
    geom_IS_list, geom_FS_list = [], []

    reaction_ids = []
    substrates, rxn_types = [], []
    IS_names, FS_names = [], []
    IS_ads, FS_ads = [], []
    IS_E_mlp, FS_E_mlp = [], []
    deltaE_mlp_list, deltaE_DFT_list = [], []
    Ea_DFT_list = []

    zeolite_active_O_indices_1based = []
    alloy_active_O_indices_1based = []

    # ==================== 主循环 ====================
    for i, r in enumerate(reactions):
        rid = r["reaction_id"]
        print(f"\n[{i+1}/{N}] reaction_id={rid}: "
              f"{r['IS_name']} -> {r['FS_name']} (substrate={r['substrate']})")
        sub = r["substrate"]

        atoms_IS_full = ase.io.read(r["IS_path"], format="vasp")
        atoms_FS_full = ase.io.read(r["FS_path"], format="vasp")
        symbols_IS = atoms_IS_full.get_chemical_symbols()

        # ---------- 识别 active O / C / H / M index ----------
        active_O_index_0 = None
        main_C_index_0 = None
        override_idx_H_0 = None
        override_idx_M_0 = None

        if sub == "zeolite":
            o_index = get_active_O_index_zeolite_by_geometry(atoms_IS_full)
            if o_index is not None:
                active_O_index_0 = o_index
                zeolite_active_O_indices_1based.append(o_index + 1)
            else:
                zeolite_active_O_indices_1based.append(0)

        elif sub == "alloy":
            o_idx = [j for j, s in enumerate(symbols_IS) if s == "O"]
            if len(o_idx) == 1:
                active_O_index_0 = o_idx[0]
                alloy_active_O_indices_1based.append(active_O_index_0 + 1)
            else:
                alloy_active_O_indices_1based.append(0)
                print(f"  [警告] 合金结构 {r['IS_name']} 中 O 原子数量为 "
                      f"{len(o_idx)} (非 1)，active_O_index 置为 None。")

        elif sub == "oxide":
            O1b = r.get("O_index_1based", None)
            if O1b is not None:
                active_O_index_0 = O1b - 1
            else:
                print("  [警告] oxide 未提供 O_index_1based，将 fallback 为与 C 最近的 O。")

        elif sub == "MN4-graphene":
            C1b = r.get("C_index_1based", None)
            if C1b is not None:
                main_C_index_0 = C1b - 1
            else:
                print("  [警告] MN4-graphene 未提供 C_index_1based，将使用第一个 C。")
            o_idx = [j for j, s in enumerate(symbols_IS) if s == "O"]
            if len(o_idx) == 1:
                active_O_index_0 = o_idx[0]
            else:
                print(f"  [警告] MN4-graphene 结构 {r['IS_name']} 中 O 原子数为 "
                      f"{len(o_idx)}，期望为 1。active_O_index=None。")

        elif sub == "MOF":
            O1b = r.get("O_index_1based", None)
            C1b = r.get("C_index_1based", None)
            H1b_list = r.get("H_index_1based", None)
            M1b_list = r.get("M_index_1based", None)

            if O1b is not None:
                active_O_index_0 = O1b - 1
            else:
                print("  [警告] MOF 未提供 O_index_1based，将 fallback 为与 C 最近的 O。")

            if C1b is not None:
                main_C_index_0 = C1b - 1
            else:
                print("  [警告] MOF 未提供 C_index_1based，将使用第一个 C。")

            if H1b_list is not None:
                override_idx_H_0 = [h - 1 for h in H1b_list]
            else:
                print("  [警告] MOF 未提供 H_index_1based，将使用所有 H。")

            if M1b_list is not None:
                override_idx_M_0 = [m - 1 for m in M1b_list]
            else:
                print("  [警告] MOF 未提供 M_index_1based，将自动识别金属元素。")

        # ---------- 计算 IS / FS 几何向量 ----------
        geom_IS = compute_geom_vector_full_structure(
            atoms_IS_full,
            sub,
            r["IS_adsorbate"],
            active_O_index=active_O_index_0,
            main_C_index=main_C_index_0,
            override_idx_H=override_idx_H_0,
            override_idx_M=override_idx_M_0,
        )

        geom_FS = compute_geom_vector_full_structure(
            atoms_FS_full,
            sub,
            r["FS_adsorbate"],
            active_O_index=active_O_index_0,
            main_C_index=main_C_index_0,
            override_idx_H=override_idx_H_0,
            override_idx_M=override_idx_M_0,
        )

        # ---------- 针对沸石 CH3OH 的 d_CO_star 特例修正 ----------
        # 条件：substrate == "zeolite"，FS_adsorbate == "CH3OH"
        #       且 FS 结构中 active_idx 里的 H 数为 5
        if sub == "zeolite" and r["FS_adsorbate"] == "CH3OH":
            symbols_FS = atoms_FS_full.get_chemical_symbols()
            n_atoms_FS = len(atoms_FS_full)
            sub_idx_FS = set(get_substrate_indices_zeolite(atoms_FS_full))
            all_idx_FS = set(range(n_atoms_FS))
            active_idx_FS = sorted(list(all_idx_FS - sub_idx_FS))
            h_active_FS = [i for i in active_idx_FS if symbols_FS[i] == "H"]
            if len(h_active_FS) == 5:
                # 强制将 FS 的 d_CO_star (geom_FS[4]) 设为 1.48 Å
                geom_FS[4] = 1.48

        geom_IS_list.append(geom_IS)
        geom_FS_list.append(geom_FS)

        # ---------- 截取局部结构 + CGCNN embedding ----------
        h_IS, atoms_IS_local = encoder.encode_local(r["IS_path"], r)
        h_FS, atoms_FS_local = encoder.encode_local(r["FS_path"], r)
        IS_emb[i] = h_IS
        FS_emb[i] = h_FS

        sub_dir = os.path.join(active_site_dir, sub)
        is_out = os.path.join(sub_dir, f"{r['IS_name']}-active.vasp")
        fs_out = os.path.join(sub_dir, f"{r['FS_name']}-active.vasp")
        ase.io.write(is_out, atoms_IS_local, format="vasp")
        ase.io.write(fs_out, atoms_FS_local, format="vasp")

        # ---------- 记录能量和元信息 ----------
        reaction_ids.append(rid)
        substrates.append(sub)
        rxn_types.append(r["reaction_type"])
        IS_names.append(r["IS_name"])
        FS_names.append(r["FS_name"])
        IS_ads.append(r["IS_adsorbate"])
        FS_ads.append(r["FS_adsorbate"])

        IS_E_mlp.append(r["IS_energy_mlp"])
        FS_E_mlp.append(r["FS_energy_mlp"])
        deltaE_mlp_list.append(r["deltaE_mlp"])
        deltaE_DFT_list.append(r["deltaE_DFT"])
        Ea_DFT_list.append(r["Ea_DFT"])

    # ==================== 保存结果 ====================

    geom_IS = np.stack(geom_IS_list, axis=0)
    geom_FS = np.stack(geom_FS_list, axis=0)
    geom_delta = geom_FS - geom_IS

    np.save(os.path.join(out_dir, "geom_IS.npy"), geom_IS)
    np.save(os.path.join(out_dir, "geom_FS.npy"), geom_FS)
    np.save(os.path.join(out_dir, "geom_delta.npy"), geom_delta)

    np.save(os.path.join(out_dir, "IS_embeddings.npy"), IS_emb)
    np.save(os.path.join(out_dir, "FS_embeddings.npy"), FS_emb)

    np.save(os.path.join(out_dir, "reaction_ids.npy"),
            np.array(reaction_ids, dtype=int))
    np.save(os.path.join(out_dir, "substrates.npy"), np.array(substrates))
    np.save(os.path.join(out_dir, "reaction_types.npy"), np.array(rxn_types))
    np.save(os.path.join(out_dir, "IS_names.npy"), np.array(IS_names))
    np.save(os.path.join(out_dir, "FS_names.npy"), np.array(FS_names))
    np.save(os.path.join(out_dir, "IS_adsorbates.npy"), np.array(IS_ads))
    np.save(os.path.join(out_dir, "FS_adsorbates.npy"), np.array(FS_ads))

    np.save(os.path.join(out_dir, "IS_energy_mlp.npy"), np.array(IS_E_mlp))
    np.save(os.path.join(out_dir, "FS_energy_mlp.npy"), np.array(FS_E_mlp))
    np.save(os.path.join(out_dir, "deltaE_mlp.npy"), np.array(deltaE_mlp_list))
    np.save(os.path.join(out_dir, "deltaE_DFT.npy"), np.array(deltaE_DFT_list))
    np.save(os.path.join(out_dir, "Ea_DFT.npy"), np.array(Ea_DFT_list))

    # --- zeolite / alloy 的 active O index 统计 + 直方图 ---
    if len(zeolite_active_O_indices_1based) > 0:
        zeolite_active_O_indices_1based = np.array(
            zeolite_active_O_indices_1based, dtype=int
        )
        np.save(os.path.join(out_dir, "zeolite_active_O_indices_1based.npy"),
                zeolite_active_O_indices_1based)
        nonzero_zeolite = zeolite_active_O_indices_1based[
            zeolite_active_O_indices_1based > 0
        ]
        if len(nonzero_zeolite) > 0:
            unique, counts = np.unique(nonzero_zeolite, return_counts=True)
            plt.figure(figsize=(8, 5), dpi=300)
            plt.bar(unique, counts, width=0.6, align="center",
                    edgecolor="black")
            for x, c in zip(unique, counts):
                plt.text(x, c, str(int(c)),
                         ha="center", va="bottom", fontsize=8)
            plt.xlabel("Active O index (1-based)")
            plt.ylabel("Count")
            plt.title("Zeolite: distribution of active O indices")
            plt.xticks(unique)
            plt.tight_layout()
            hist_path = os.path.join(out_dir, "zeolite_active_O_index_hist.jpg")
            plt.savefig(hist_path, dpi=300)
            plt.close()
            print(f"沸石活性 O 直方图已保存到: {hist_path}")
        else:
            print("警告：沸石结构中未找到任何活性 O（全部为 0），"
                  "未绘制 zeolite_active_O_index_hist.jpg。")

    if len(alloy_active_O_indices_1based) > 0:
        alloy_active_O_indices_1based = np.array(
            alloy_active_O_indices_1based, dtype=int
        )
        np.save(os.path.join(out_dir, "alloy_active_O_indices_1based.npy"),
                alloy_active_O_indices_1based)
        nonzero_alloy = alloy_active_O_indices_1based[
            alloy_active_O_indices_1based > 0
        ]
        if len(nonzero_alloy) > 0:
            unique_a, counts_a = np.unique(nonzero_alloy, return_counts=True)
            plt.figure(figsize=(8, 5), dpi=300)
            plt.bar(unique_a, counts_a, width=0.6,
                    align="center", edgecolor="black")
            for x, c in zip(unique_a, counts_a):
                plt.text(x, c, str(int(c)),
                         ha="center", va="bottom", fontsize=8)
            plt.xlabel("Active O index (1-based)")
            plt.ylabel("Count")
            plt.title("Alloy: distribution of active O indices")
            plt.xticks(unique_a)
            plt.tight_layout()
            hist_path_a = os.path.join(out_dir, "alloy_active_O_index_hist.jpg")
            plt.savefig(hist_path_a, dpi=300)
            plt.close()
            print(f"合金主 O 直方图已保存到: {hist_path_a}")
        else:
            print("警告：合金结构中未找到任何主 O（全部为 0），"
                  "未绘制 alloy_active_O_index_hist.jpg。")

    # --- 总结 ---
    print(f"\n所有数据已保存到 {out_dir}/")
    print("geom 的 9 维含义：")
    print(" 0: d_CM_min (C-M 最近距离, PBC)")
    print(" 1: CN_CH (C-H 配位数, d_CH < 1.3 Å, PBC)")
    print(" 2: max_d_CH (C-H 最大距离, PBC；沸石 CH4 有特例 1.10 Å)")
    print(" 3: CN_CO (C-O 配位数, d_CO < 1.6 Å, PBC)")
    print(" 4: d_CO_star (C-主 O* 距离, PBC；沸石 CH3OH 有特例 1.48 Å)")
    print(" 5: d_OM_min (主 O*-M 最近距离, PBC)")
    print(" 6: CN_M (主 O*-M 配位数, d_OM < 3.0 Å, PBC)")
    print(" 7: CN_OH (O-H 配位数, d_OH < 1.3 Å, PBC)")
    print(" 8: min_d_OH (主 O*-所有 H 的最近距离, PBC)")
    print("同时保存了 IS/FS 的 MLP 能量、ΔE_mlp、ΔE_DFT、Ea_DFT 等，")
    print("以及 IS/FS 的 CGCNN 局部 embedding 和 active_site VASP 。")


# ======================================================================
# 5. main
# ======================================================================

if __name__ == "__main__":
    set_seed(42)
    device = "cpu"  # 如有 GPU 可改 "cuda"

    csv_path = "879-9762-0330-reaction-list.csv"
    prepare_Ea_reg_all(csv_path=csv_path, out_dir="embeddings", device=device)