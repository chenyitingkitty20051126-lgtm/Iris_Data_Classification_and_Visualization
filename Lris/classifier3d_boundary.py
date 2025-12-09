import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# --- 1. 数据准备与转换 ---
iris = load_iris()

# 选择特征：x1 (Sepal Width), x2 (Petal Length), x3 (Petal Width)
X_all = iris.data[:, 1:4] 
y_all = iris.target

# 转换为二分类问题: 只保留类别 1 (Versicolor) 和 类别 2 (Virginica)
X = X_all[y_all > 0]
y = y_all[y_all > 0]
y[y == 1] = 0 # Versicolor 标记为 0 (蓝色)
y[y == 2] = 1 # Virginica 标记为 1 (红色)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. 模型训练 ---
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_scaled, y)

# 获取模型的参数 (w 和 b)
w1, w2, w3 = model.coef_[0]
b = model.intercept_[0]

# --- 3. 计算超平面上的 Z 值 ---
x1_min, x1_max = X_scaled[:, 0].min() - 0.2, X_scaled[:, 0].max() + 0.2
x2_min, x2_max = X_scaled[:, 1].min() - 0.2, X_scaled[:, 1].max() + 0.2
XX1, XX2 = np.meshgrid(np.linspace(x1_min, x1_max, 10), # 减少网格密度，提高速度
                       np.linspace(x2_min, x2_max, 10))

# 计算 X3 (Z) 的值
if w3 != 0:
    XX3 = -(w1 * XX1 + w2 * XX2 + b) / w3
else:
    raise ValueError("The coefficient for the third feature is too close to zero.")

# --- 4. 3D 可视化：美化部分 ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 颜色设置
data_colors = np.array(['#1f77b4', '#d62728'])[y] # 蓝色和红色，更鲜明
plane_color = '#aaaaaa' # 浅灰色

# 绘制散点图
# c=data_colors: 使用数组颜色
# s=80: 增大点的大小
# alpha=0.9: 提高不透明度
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
           c=data_colors, marker='o', edgecolors='k', s=80, alpha=0.9)

# 绘制决策超平面
# alpha=0.7: 适当调高透明度
# cmap=plt.cm.coolwarm: 使用 colormap 可以让平面有渐变效果（可选）
ax.plot_surface(XX1, XX2, XX3, 
                alpha=0.6, # 比数据点透明一些
                color=plane_color,
                rstride=1, cstride=1, # 保持网格密度，但使用 linewidths=0 移除线条
                linewidths=0, antialiased=False)

# 优化坐标轴和网格线
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # 移除背景面板颜色
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.grid(True, linestyle='--', alpha=0.5) # 使用虚线网格

# 设置标签和标题 (使用 LaTex 格式)
ax.set_xlabel('$X_1$: Sepal Width (Scaled)', fontsize=12)
ax.set_ylabel('$X_2$: Petal Length (Scaled)', fontsize=12)
ax.set_zlabel('$X_3$: Petal Width (Scaled)', fontsize=12)
ax.set_title('3D Linear Decision Hyperplane (Logistic Regression)', fontsize=14)

# 调整视角 (可以根据需要调整 elev 和 azim)
ax.view_init(elev=20, azim=130)

# 移除图例（因为颜色在标题中说明）
# fig.legend(...) 

plt.show()