import numpy as np
import os

# 路径
base_dir = "embeddings"
geom_file = os.path.join(base_dir, "geom_delta_080_4_fea.npy")
deltaE_file = os.path.join(base_dir, "deltaE_mlp.npy")
out_csv = os.path.join(base_dir, "geom_delta-deltaE_mlp_080_4_fea.csv")

# 1. 读取数据
geom_delta = np.load(geom_file)        # shape: (9762, 9)
deltaE_mlp = np.load(deltaE_file)      # shape: (9762,)

# 3. 将 deltaE_mlp 变成列向量，并与 geom_delta 按列拼接
deltaE_mlp_col = deltaE_mlp.reshape(-1, 1)     # (9762, 1)
combined = np.hstack([geom_delta, deltaE_mlp_col])  # (9762, 10)

# 4. 保存为 CSV 文件（不带表头，如需表头可自行添加）
np.savetxt(out_csv, combined, delimiter=",")
print(f"已保存到: {out_csv}")
