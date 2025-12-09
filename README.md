# Lris
🌸 Project 3: 鸢尾花数据分类与可视化 (Iris Classification and Visualization)使用 Python + scikit-learn + Plotly 交互式可视化 开发的机器学习模型探索项目。功能 Features本项目通过四个核心任务，系统地展示了机器学习模型从 2D 到 3D 的决策过程和概率分布。任务编号功能模块描述Task 0数据探索 (EDA)使用 Seaborn 分析特征分布，Plotly 绘制交互式散点图。Task 12D 边界与概率在 2D 平面（花瓣长/宽）上，对比 LogReg 和 GPC 等模型的决策边界和概率热力图。Task 23D 线性超平面在 3D 空间中，可视化 LogReg 模型对二分类问题的线性决策超平面。Task 33D 概率曲面绘制 GPC 模型的非线性概率曲面，并将其完全投影至 XOY/YOZ/ZOX 边界平面。Task 43D 决策区域渲染利用 Plotly Volume 迹线，渲染 GPC 模型对三分类问题的三维决策区域块。程序架构（Illustration 图示）本项目的架构展示了数据流从数据探索到高维模型可视化逐步递进的过程。📂 项目结构.
├── data_preview.py             # 任务 0: 数据探索分析
├── classifier2d_initial.py     # 任务 1: 2D 决策边界 (LogReg 基础)
├── classifier2d.py             # 任务 1: 2D 决策边界与概率图 (多模型对比)
├── classifier3d_boundary.py    # 任务 2: 3D 线性决策超平面
├── classifier3d_probability.py # 任务 3: 3D 概率曲面与投影
└── classifier3d_volume.py      # 任务 4: 3D 决策区域体积渲染
运行方法（Run）首先，确保安装了所有必要的依赖库：Bash# 核心库：scikit-learn, numpy, pandas
# 可视化库：matplotlib, seaborn, plotly, scikit-image
pip install numpy pandas scikit-learn matplotlib seaborn plotly scikit-image
运行主程序（例如运行任务四，查看最终 3D 效果）：Bashpython classifier3d_volume.py
