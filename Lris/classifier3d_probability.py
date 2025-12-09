import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
from skimage import measure 
import warnings
warnings.filterwarnings("ignore") 

# --- 1. 数据准备与模型训练 ---
iris = load_iris()
# 选择特征：Petal Length (x2) 和 Petal Width (x3)
X_all = iris.data[:, 2:]  
y_all = iris.target
X = X_all[y_all > 0]
y = y_all[y_all > 0]
y[y == 1] = 0 # Versicolor 
y[y == 2] = 1 # Virginica 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kernel = 1.0 * RBF(length_scale=[0.5, 0.5]) 
model = GaussianProcessClassifier(kernel=kernel, random_state=42)
model.fit(X_scaled, y)

# --- 2. 概率曲面和网格计算 ---
x_min, x_max = X_scaled[:, 0].min() - 0.2, X_scaled[:, 0].max() + 0.2
y_min, y_max = X_scaled[:, 1].min() - 0.2, X_scaled[:, 1].max() + 0.2
z_min, z_max = 0, 1.0

step = 0.02
XX1, XX2 = np.meshgrid(np.arange(x_min, x_max, step), 
                       np.arange(y_min, y_max, step))

X_grid = np.c_[XX1.ravel(), XX2.ravel()]
Z_prob_surface = model.predict_proba(X_grid)[:, 1].reshape(XX1.shape)
Z_prob_data = model.predict_proba(X_scaled)[:, 1]

# 定义配色方案
df_data = pd.DataFrame(X_scaled, columns=['X2_Petal_Length', 'X3_Petal_Width'])
df_data['target'] = y
data_colors = np.where(df_data['target'] == 1, 'red', 'blue') 
custom_colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]

# =========================================================================
# A. 3D 可视化 (将投影不透明度设置为 1.0，并添加网格线)
# =========================================================================

fig_3d = go.Figure()

# 1. 绘制 3D 概率曲面 (主体) - 适当降低透明度，避免遮挡投影
fig_3d.add_trace(go.Surface(
    x=XX1, y=XX2, z=Z_prob_surface, 
    colorscale=custom_colorscale, 
    opacity=0.7, # 主曲面透明度略降
    name='P(Class=1) Surface',
    # 移除主曲面上的网格线（如果需要），使其更平滑，但保留颜色
    contours_z=dict(show=False), 
    showscale=False,
    lighting=dict(ambient=0.8, diffuse=0.1, roughness=0.5, specular=0.1)
))

# 2. 绘制原始数据点
fig_3d.add_trace(go.Scatter3d(
    x=df_data['X2_Petal_Length'], 
    y=df_data['X3_Petal_Width'], 
    z=Z_prob_data, 
    mode='markers',
    marker=dict(
        size=4, 
        color=data_colors,
        opacity=1, 
        line=dict(width=1, color='Black')
    ),
    name='Data Points'
))

# --- 3. 底部填充投影 (X2 vs X3) - XOY 平面 (Z=0) ---
fig_3d.add_trace(go.Surface(
    x=XX1, y=XX2, z=np.full_like(XX1, z_min), 
    surfacecolor=Z_prob_surface, 
    colorscale=custom_colorscale,
    opacity=1.0, # 完全不透明
    showscale=False, name='Bottom P Contour (XOY)',
    # 添加网格线
    contours_z=dict(show=True, project_z=True, color='gray', width=1, highlightcolor="gray"), 
))

# --- 4. 侧面填充投影 1 (X2 vs P) - ZOX 平面 (Y=y_max) ---
fig_3d.add_trace(go.Surface(
    x=XX1, y=np.full_like(XX1, y_max), z=Z_prob_surface, 
    surfacecolor=Z_prob_surface, 
    colorscale=custom_colorscale,
    opacity=1.0, # 完全不透明
    showscale=False, name='Side P Contour (ZOX)',
    # 添加网格线
    contours_y=dict(show=True, project_y=True, color='gray', width=1, highlightcolor="gray"),
))

# --- 5. 侧面填充投影 2 (X3 vs P) - YOZ 平面 (X=x_min) ---
fig_3d.add_trace(go.Surface(
    x=np.full_like(XX2, x_min), y=XX2, z=Z_prob_surface, 
    surfacecolor=Z_prob_surface, 
    colorscale=custom_colorscale,
    opacity=1.0, # 完全不透明
    showscale=False, name='Side P Contour (YOZ)',
    # 添加网格线
    contours_x=dict(show=True, project_x=True, color='gray', width=1, highlightcolor="gray"),
))


fig_3d.update_layout(
    title='Task 3 (1/4): 3D Probability Map with Fully Opaque Contour Projections',
    scene=dict(
        xaxis_title='X2: Petal Length (Scaled)',
        yaxis_title='X3: Petal Width (Scaled)',
        zaxis_title='Probability P(Class=1)', 
        zaxis_range=[z_min, z_max + 0.1], 
        # 调整坐标轴的网格线颜色，使其与投影网格区分
        xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray", showbackground=True),
        yaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray", showbackground=True),
        zaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray", showbackground=True),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.5) 
    ),
    margin=dict(r=0, l=0, b=0, t=30)
)
fig_3d.show()


# =========================================================================
# B. 三个 2D 投影图 (保持不变)
# =========================================================================

# 创建一个 1 行 3 列的子图布局
fig_2d = make_subplots(
    rows=1, cols=3,
    subplot_titles=("2D Projection 1: X2 vs X3 (Feature Space)", 
                    "2D Projection 2: X2 vs P (Marginal Probability)", 
                    "2D Projection 3: X3 vs P (Marginal Probability)")
)

# --- 1. X2 vs X3 投影图（特征空间） ---
fig_2d.add_trace(go.Contour(
    x=XX1[0, :], y=XX2[:, 0], z=Z_prob_surface, 
    colorscale=custom_colorscale, 
    contours_coloring='heatmap',
    showscale=False,
    name='P(X2, X3) Contour'
), row=1, col=1)

# 添加数据点到 X2 vs X3 图中
fig_2d.add_trace(go.Scatter(
    x=df_data['X2_Petal_Length'], y=df_data['X3_Petal_Width'], 
    mode='markers',
    marker=dict(size=7, color=data_colors, line=dict(width=1, color='black')),
    showlegend=True, name='Data Points (X2, X3)'
), row=1, col=1)

# --- 2. X2 vs P 投影图（P 随 X2 的变化）---
center_index = Z_prob_surface.shape[0] // 2 

fig_2d.add_trace(go.Scatter(
    x=XX1[center_index, :], y=Z_prob_surface[center_index, :], 
    mode='lines',
    line=dict(color='black', width=3),
    name='P(X2) Curve (Approx)'
), row=1, col=2)

# 添加数据点到 X2 vs P 图中
fig_2d.add_trace(go.Scatter(
    x=df_data['X2_Petal_Length'], y=Z_prob_data, 
    mode='markers',
    marker=dict(size=7, color=data_colors, line=dict(width=1, color='black')),
    showlegend=False
), row=1, col=2)

# --- 3. X3 vs P 投影图（P 随 X3 的变化）---
center_index = Z_prob_surface.shape[1] // 2 

fig_2d.add_trace(go.Scatter(
    x=XX2[:, center_index], y=Z_prob_surface[:, center_index], 
    mode='lines',
    line=dict(color='black', width=3),
    name='P(X3) Curve (Approx)'
), row=1, col=3)

# 添加数据点到 X3 vs P 图中
fig_2d.add_trace(go.Scatter(
    x=df_data['X3_Petal_Width'], y=Z_prob_data, 
    mode='markers',
    marker=dict(size=7, color=data_colors, line=dict(width=1, color='black')),
    showlegend=False
), row=1, col=3)


# 调整 2D 图布局
fig_2d.update_layout(title_text="Task 3 (2/4 - 4/4): 2D Projections of Probability Map", height=450)
fig_2d.update_xaxes(title_text="X2: Petal Length (Scaled)", row=1, col=1)
fig_2d.update_yaxes(title_text="X3: Petal Width (Scaled)", row=1, col=1)

fig_2d.update_xaxes(title_text="X2: Petal Length (Scaled)", row=1, col=2)
fig_2d.update_yaxes(title_text="Probability P(Class=1)", range=[z_min, z_max + 0.1], row=1, col=2)

fig_2d.update_xaxes(title_text="X3: Petal Width (Scaled)", row=1, col=3)
fig_2d.update_yaxes(title_text="Probability P(Class=1)", range=[z_min, z_max + 0.1], row=1, col=3)

fig_2d.show()