import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
# 不再需要 skimage.measure

# --- 1. 数据准备与模型训练 ---
iris = load_iris()
# 选择 X1: Sepal Width, X2: Petal Length, X3: Petal Width 作为特征
X = iris.data[:, 1:4] 
y = iris.target        
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 模型训练：使用非线性分类器 (高斯过程 GPC)
kernel = 1.0 * RBF(length_scale=[1.0, 1.0, 1.0]) 
model = GaussianProcessClassifier(kernel=kernel, random_state=42)
model.fit(X_scaled, y)

# --- 2. 决策区域网格计算 ---
# 网格密度 (50x50x50)
grid_points = 50 
x1_min, x1_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
x2_min, x2_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
x3_min, x3_max = X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5

X_coords = np.linspace(x1_min, x1_max, grid_points)
Y_coords = np.linspace(x2_min, x2_max, grid_points)
Z_coords = np.linspace(x3_min, x3_max, grid_points)
XX1, XX2, XX3 = np.meshgrid(X_coords, Y_coords, Z_coords, indexing='ij')

X_grid = np.c_[XX1.ravel(), XX2.ravel(), XX3.ravel()]
Z_pred_volume = model.predict(X_grid) # 预测类别 (0, 1, 2)
Z_pred_volume = Z_pred_volume.reshape((grid_points, grid_points, grid_points))

# --- 3. Plotly 3D 可视化 ---
fig = go.Figure()

# 3.1 定义颜色映射：将离散的类别 (0, 1, 2) 映射到固定的颜色 (黄、绿、蓝)
# Colorscale 定义了值和颜色：[0.0到0.33是黄], [0.33到0.66是绿], [0.66到1.0是蓝]
colorscale = [
    [0.0, 'yellow'],  # 类别 0: Setosa
    [0.33, 'yellow'], 
    [0.33, 'green'], 
    [0.66, 'green'],  # 类别 1: Versicolor
    [0.66, 'blue'],
    [1.0, 'blue']    # 类别 2: Virginica
]

# 3.2 绘制决策区域 (使用 Volume Trace)
# volume 迹线将 Z_pred_volume (类别标签) 渲染为 3D 颜色块
fig.add_trace(go.Volume(
    x=XX1.flatten(), y=XX2.flatten(), z=XX3.flatten(), 
    value=Z_pred_volume.flatten(),
    isomin=0,  # 类别最小值
    isomax=2,  # 类别最大值
    opacity=0.2,      # 设置半透明度，让内部的数据点可见
    surface_count=3,  # 渲染 3 个独立的颜色区域
    colorscale=colorscale,
    showscale=False,
    name='Decision Regions'
))

# 3.3 绘制原始数据点 
df_data = pd.DataFrame(X_scaled, columns=['X1', 'X2', 'X3'])
df_data['target'] = y
color_map = {0: 'yellow', 1: 'green', 2: 'blue'} 
df_data['Color_data'] = df_data['target'].map(color_map)

fig.add_trace(go.Scatter3d(
    x=df_data['X1'], y=df_data['X2'], z=df_data['X3'],
    mode='markers',
    marker=dict(
        size=8,
        color=df_data['Color_data'],
        opacity=1,
        line=dict(width=1.5, color='Black')
    ),
    name='Data Points'
))

# 3.4 布局设置
fig.update_layout(
    title='Task 4: 3D Non-Linear Decision Regions (GPC) - Solid Color Blocks',
    scene=dict(
        xaxis_title='X1: Sepal Width (Scaled)',
        yaxis_title='X2: Petal Length (Scaled)',
        zaxis_title='X3: Petal Width (Scaled)',
        aspectmode='cube'
    ),
    margin=dict(r=0, l=0, b=0, t=30)
)

fig.show()