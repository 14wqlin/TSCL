# -*- coding: utf-8 -*-
"""
extract_4_geom_features.py

功能：
  从 all_geom_site_embedding.py 生成的 embeddings/geom_delta.npy 中，
  提取编号为 1,4,5,6 的四个几何特征列：
      1: CN_CH
      4: d_CO_star
      5: d_OM_min
      6: CN_M
  并保存为新的几何向量文件 geom_delta_080_4_fea.npy
"""

import os
import numpy as np

# 如果脚本与 embeddings 在同一级目录，默认路径如下；
# 如果脚本放在别处，请把 base_dir 改成你的 embeddings 路径。
base_dir = "embeddings"
in_path = os.path.join(base_dir, "geom_delta.npy")
out_path = os.path.join(base_dir, "geom_delta_080_4_fea.npy")

# 1. 读取原始 9 维几何向量
geom_delta = np.load(in_path)  # 形状 (N, 9)
print("原始 geom_delta 形状:", geom_delta.shape)

# 2. 选择需要的列：1,4,5,6（从 0 开始编号）
#    对应特征：1: CN_CH, 4: d_CO_star, 5: d_OM_min, 6: CN_M
selected_indices = [1, 4, 5, 6]
geom_delta_4 = geom_delta[:, selected_indices]
print("筛选后的 4 维几何向量形状:", geom_delta_4.shape)

# 3. 保存为新的 npy 文件
np.save(out_path, geom_delta_4)
print(f"已将 4 维几何向量保存为: {out_path}")

