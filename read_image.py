from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 读取图片，得到X =====================
im = Image.open(r'E:\PythonProject5\data\flower.jpg')
image = np.array(im).astype(float) / 255
image = image[:, :, :3]
H, W, C = image.shape
X = image.reshape(-1, 3)
print("X shape:", X.shape)


# ===================== 2. 手写K-Means函数 myKmeans【作业标准版本】 =====================
def myKmeans(X, K, random_seed):
    np.random.seed(random_seed)
    n_samples = X.shape[0]

    # 第一步：初始化簇中心 initCenter，随机挑选K个样本
    idx = np.random.choice(n_samples, K, replace=False)
    classCenter = X[idx].copy()

    max_iter = 100  # 防止死循环，设置最大迭代次数
    prev_cluster_idx = None  # 保存上一轮簇分配结果

    for _ in range(max_iter):
        # 第二步：分配样本，向量化计算【欧氏距离平方】
        dist_sq = np.sum((X[:, np.newaxis] - classCenter) ** 2, axis=2)  # shape(N, K)
        cluster_idx = np.argmin(dist_sq, axis=1)  # 每个样本归属簇编号 shape(N,)

        # 构造 clusterPoints: (样本数,2)，第0列簇号，第1列距离平方
        sample_dist_sq = dist_sq[np.arange(n_samples), cluster_idx]
        clusterPoints = np.column_stack([cluster_idx, sample_dist_sq])

        # 判断收敛：本轮簇分配和上一轮完全一样，停止迭代
        if prev_cluster_idx is not None and np.all(cluster_idx == prev_cluster_idx):
            break
        prev_cluster_idx = cluster_idx.copy()

        # 第三步：更新簇中心
        new_centers = np.zeros_like(classCenter)
        for k in range(K):
            points_in_cluster = X[cluster_idx == k]
            if len(points_in_cluster) > 0:
                new_centers[k] = points_in_cluster.mean(axis=0)
            else:
                # 空簇：保留原来中心不变
                new_centers[k] = classCenter[k]
        classCenter = new_centers
    return classCenter, clusterPoints


# ===================== 3. 肘方法：K=1~16，每个K跑3个种子 =====================
K_list = list(range(1, 17))
best_sse_for_K = []
seeds = [10, 42, 99]  # 3个不同随机种子

for K in K_list:
    sse_list = []
    for seed in seeds:
        classCenter, clusterPoints = myKmeans(X, K, seed)
        sse = clusterPoints[:, 1].sum()
        sse_list.append(sse)
        print(f"K={K}, seed={seed}, SSE={sse:.2f}")
    min_sse = np.min(sse_list)
    best_sse_for_K.append(min_sse)
    print(f"---- K={K} 最小SSE = {min_sse:.2f}\n")

# ===================== 打印SSE下降量 =====================
print("==== SSE下降量 ====")
for i in range(1, len(K_list)):
    drop = best_sse_for_K[i-1] - best_sse_for_K[i]
    print(f"K={K_list[i-1]} → K={K_list[i]}，SSE下降: {drop:.2f}")

# ===================== 绘制肘曲线 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,5))
plt.plot(K_list, best_sse_for_K, marker='o', linestyle='-')
plt.xlabel("聚类数量 K")
plt.ylabel("最小SSE（簇内距离平方和）")
plt.title("K-Means 肘方法确定最优K")
plt.xticks(K_list)
plt.grid(True)
plt.show()


# ===================== 4. 矢量量化重建与展示【第4部分】 =====================
# --------------------------【重要！！！】--------------------------
# ！！！这里把best_k改成你从肘曲线图观察得到的肘部K值！！！
best_k = 5
# ----------------------------------------------------------------
test_Ks = [2, best_k, 30]
reconstruct_imgs = []
titles = ["原图"]

# 原图放入列表
reconstruct_imgs.append(image)

# 分别运行myKmeans重建图片
for k in test_Ks:
    cc, cp = myKmeans(X, k, random_seed=42)
    c_idx = cp[:,0].astype(int)
    # 像素替换为簇中心颜色
    recon_X = cc[c_idx]
    recon_img = recon_X.reshape(H, W, 3)
    reconstruct_imgs.append(recon_img)
    titles.append(f"K={k}")

# 并排绘制 原图，K=2，最优K，K=30
plt.figure(figsize=(16, 5))
for i, img in enumerate(reconstruct_imgs):
    plt.subplot(1, 4, i+1)
    plt.imshow(img)
    plt.title(titles[i])
    plt.axis("off")
plt.suptitle("原图与不同K矢量量化重建图像对比", fontsize=14)
plt.tight_layout()
plt.show()
