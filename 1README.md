# GeomPrompt Extractor: 几何感知前景提取工具

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-Apache_2.0-orange.svg)

**GeomPrompt Extractor** 是一个基于 Meta AI 的 [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 开发的先进前景提取工具。

本项目针对**木雕、浮雕、金银花纹、植物**等具有复杂几何形状和精细纹理的对象进行了深度优化，解决了通用分割工具在这些场景下边缘提取不准、主体识别错误、细节丢失等痛点。

---

## 核心特性

- **✨ 智能对象类型识别**: 自动分析图像内容，将其分类为雕刻品 (`CARVING`)、浮雕 (`RELIEF`)、花草 (`FLOWER`) 等类型，并为不同类型启用专属的优化处理流程。

- **🎯 增强多策略内容定位**: 融合显著性检测、自适应GrabCut、分层边缘检测、颜色聚类等六种策略，精准定位核心目标区域，为SAM提供高质量的输入。

- **📐 独创几何感知点生成**: 摒弃传统的随机点或网格点提示，本工具从初始掩码中提取**角点、高曲率点、轮廓关键点、高梯度点**等几何特征，生成对物体形状感知能力更强的提示点，极大提升了SAM在复杂边缘上的分割精度。

- **🚀 多提示（Multi-Prompt）分割策略**: 结合内容框、几何感知点集和初始掩码，以多种方式向SAM提问，并从中选择质量最高的分割结果。

- **🔧 自适应后处理**: 根据对象类型，对分割掩码进行定制化的形态学操作、孔洞填充和边缘平滑，确保最终前景的完整性与平滑度。

- **⚙️ 高效批量处理与实验管理**: 内置批量处理、超时保护和内存管理机制，并支持通过Git分支和日志文件进行可复现的实验跟踪。

## 效果演示

*(建议在此处替换为你自己的项目效果图)*

| 原始图像 | 提取结果 |
| :---: | :---: |
| <img src="docs/assets/carving_before.jpg" width="300"/> | <img src="docs/assets/carving_after.png" width="300"/> |
| <img src="docs/assets/relief_before.jpg" width="300"/> | <img src="docs/assets/relief_after.png" width="300"/> |

## 安装指南

**环境要求**:
* Python 3.9+
* PyTorch & TorchVision
* Git

**安装步骤**:

1.  **克隆本项目**
    ```bash
    git clone <你的项目Git仓库地址>
    cd CarvingExtractor
    ```

2.  **创建并激活Python虚拟环境**
    ```bash
    # 创建虚拟环境
    python -m venv venv

    # 激活虚拟环境
    # Windows:
    # venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    ```

3.  **安装项目依赖**
    本项目所有依赖（包括`segment-anything`库）都已在 `requirements.txt` 中定义。
    ```bash
    pip install -r requirements.txt
    ```

4.  **下载模型权重文件**
    请从以下链接下载 SAM ViT-H 模型，并将其放置在项目的根目录下。
    
    [**`sam_vit_h_4b8939.pth`**](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (约2.4GB)
    
    下载后，请确保根目录结构如下:
    ```
    CarvingExtractor/
    ├── sam_vit_h_4b8939.pth  <-- 模型文件在此
    ├── main.py
    └── ...
    ```

## 使用方法

1.  **准备输入图像**
    将需要处理的图像文件放入 `input_images` 文件夹内。

2.  **运行主程序**
    在项目根目录打开终端，直接运行 `main.py` 即可开始批量处理。
    ```bash
    python main.py
    ```

3.  **查看结果**
    * 处理结果将默认保存在 `outputs/<时间戳_实验名>` 文件夹中。
    * `*_foreground.png`: 提取出的前景图像（透明背景）。
    * `*_v8_analysis.png`: 包含内容框、初始掩码、最终结果等信息的可视化分析图。
    * 终端会输出每次处理的详细日志以及最终的质量评估报告。

## 项目结构

```
CarvingExtractor/
├── input_images/               # 存放待处理的原始图像
├── outputs/                    # 存放所有处理结果
├── geom_prompt_extractor/      # 项目核心源代码包
│   ├── __init__.py
│   ├── config.py               # 核心参数配置
│   ├── sam_core.py             # Part 1: SAM模型与内容检测
│   ├── segmentation_processor.py # Part 2: 几何点生成与分割
│   └── post_processor.py       # Part 3: 后处理与流程编排
├── main.py                     # 主程序入口
├── requirements.txt            # 项目依赖列表
├── experiments.md              # 实验记录日志
└── sam_vit_h_4b8939.pth        # SAM模型权重
```

## 致谢

本项目基于 Meta AI Research 的 [Segment Anything](https://github.com/facebookresearch/segment-anything) 项目。感谢其为计算机视觉社区提供的强大基础模型。
