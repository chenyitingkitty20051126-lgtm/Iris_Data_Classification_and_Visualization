import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- 1. 数据准备 ---
iris = load_iris()
# 任务一使用后两个特征：花瓣长度 (Petal Length) 和花瓣宽度 (Petal Width)
X = iris.data[:, 2:] 
y = iris.target

# 为了更好地比较不同模型的决策边界，我们对特征进行标准化
X = StandardScaler().fit_transform(X)

# 划分数据集 (可选，但推荐用于更真实的评估)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# --- 2. 定义多个分类器 (参考 PPT Slide 5) ---
classifiers = {
    # 逻辑回归 (低正则化)
    "Logistic Regression (C=0.1)": LogisticRegression(C=0.1, max_iter=200, random_state=42),
    # 逻辑回归 (中等正则化)
    "Logistic Regression (C=1)": LogisticRegression(C=1, max_iter=200, random_state=42),
    # 高斯过程
    "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0]), random_state=42),
    # 梯度提升
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
}

# --- 3. 网格点创建 ---
# 创建网格以绘制决策区域和概率图
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 用于散点图的颜色映射
class_colors = ['yellow', 'green', 'blue']  
cmap_data = mcolors.ListedColormap(class_colors)


# --- 4. 循环训练和可视化 ---

# 设置整体大图的尺寸
fig, axes = plt.subplots(len(classifiers), 4, figsize=(20, 5 * len(classifiers))) 
if len(classifiers) == 1: # 处理只有一个分类器的情况
    axes = np.expand_dims(axes, axis=0)

# 遍历每个分类器
for i, (name, classifier) in enumerate(classifiers.items()):
    
    # 训练模型
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    
    # 在网格点上计算预测结果和概率
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(X_grid).reshape(xx.shape)
    probs = classifier.predict_proba(X_grid).reshape(xx.shape[0], xx.shape[1], 3) 
    
    # 设置当前行
    row_axes = axes[i]
    
    # --- 第1列: 整体决策边界图 ---
    ax_boundary = row_axes[0]
    
    # 使用 contourf 绘制决策区域
    ax_boundary.contourf(xx, yy, Z, cmap=cmap_data, alpha=0.6)
    
    # 绘制数据点
    ax_boundary.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=cmap_data)
    
    ax_boundary.set_title(f'{name}\nOverall Decision Boundaries (Score: {score:.2f})')
    ax_boundary.set_xlabel('Petal Length (X2)')
    ax_boundary.set_ylabel('Petal Width (X3)')
    
    # --- 第2-4列: 每一类的概率图 ---
    for j, class_prob in enumerate(probs.transpose(2, 0, 1)):
        ax_prob = row_axes[j + 1] 
        
        # 定义从白色到类别色的渐变色图
        cmap_prob = mcolors.LinearSegmentedColormap.from_list(
            f'class_{j}_colormap', ['white', class_colors[j]], N=256)
        
        # 绘制概率等高线图
        contour = ax_prob.contourf(xx, yy, class_prob, levels=np.linspace(0, 1, 11), alpha=0.7, cmap=cmap_prob)
        
        # 绘制数据点
        ax_prob.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=cmap_data, alpha=1)
        
        # 添加 color bar
        fig.colorbar(contour, ax=ax_prob, ticks=np.linspace(0, 1, 6))
        
        # 设置标题和标签
        ax_prob.set_title(f'{name}\nClass {j} Probability')
        ax_prob.set_xlabel('Petal Length (X2)')
        ax_prob.set_ylabel('Petal Width (X3)')
        
# 调整布局
plt.tight_layout()
plt.show()