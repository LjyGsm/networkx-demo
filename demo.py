# -*- coding: utf-8 -*-
"""
NetworkX 上手演示：ER 随机网络 vs BA 无标度网络
- 生成网络
- 统计度分布并画图
- 计算平均聚类系数
运行：python demo.py  （先激活 .venv）
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

N = 1000  # 节点数


def degree_dist(G, ax, title):
    """画度分布直方图"""
    degrees = np.array([d for _, d in G.degree()])
    ax.hist(degrees, bins=30, density=True, edgecolor="black")
    ax.set_title(f"{title}\navg degree = {degrees.mean():.2f}, "
                 f"C = {nx.average_clustering(G):.4f}")
    ax.set_xlabel("degree k")
    return degrees


# ---------- 1. 生成网络 ----------
ER = nx.erdos_renyi_graph(N, 0.01)        # ER 随机网络 G(N, p)
BA = nx.barabasi_albert_graph(N, 3)       # BA 无标度网络（每次新节点接 3 条边）

# ---------- 2. 基本统计 ----------
for name, G in [("ER", ER), ("BA", BA)]:
    degs = [d for _, d in G.degree()]
    print(f"{name}: 平均度 = {np.mean(degs):.2f}, "
          f"平均聚类系数 = {nx.average_clustering(G):.4f}, "
          f"边数 = {G.number_of_edges()}")

# 度数前 5 的枢纽（BA 网络里应该很明显）
print("BA 度数前 5 的节点:",
      sorted(nx.degree(BA), key=lambda x: -x[1])[:5])

# ---------- 3. 度分布对比图（线性坐标 + BA 的 log-log） ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
degree_dist(ER, axes[0], "ER (N=1000, p=0.01)")
degrees_ba = degree_dist(BA, axes[1], "BA (N=1000, m=3)")
axes[0].set_ylabel("P(k)")

# BA 的 log-log 图：幂律分布近似一条直线
bins = np.logspace(np.log10(max(1, degrees_ba.min())),
                   np.log10(degrees_ba.max()), 20)
hist, edges = np.histogram(degrees_ba, bins=bins, density=True)
centers = np.sqrt(edges[:-1] * edges[1:])
axes[2].loglog(centers, hist, "o-")
axes[2].set_title("BA degree dist (log-log)")
axes[2].set_xlabel("degree k")
axes[2].set_ylabel("P(k)")

plt.tight_layout()
plt.savefig("degree_dist.png", dpi=150)
print("图已保存 -> degree_dist.png")
