---

### 文件 3: `README_zh.md` (中文版文件 - 最终版)

* **操作**：请创建 `README_zh.md` 文件，并将以下所有内容复制进去。
* **说明**：修正了所有格式问题，并添加了缺失的链接。

```markdown
[English Version](README.md)

# GeomPrompt Extractor: 几何感知前景提取工具

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache_2.0-orange.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2403.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2403.XXXXX) **GeomPrompt Extractor** 是一个基于 Meta AI 的 [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 开发的先进前景提取工具。

本项目针对**木雕、浮雕、金银花纹、植物**等具有复杂几何形状和精细纹理的对象进行了深度优化，解决了通用分割工具在这些场景下边缘提取不准、主体识别错误、细节丢失等痛点。它是一个强大的工具，可用于学术研究、数字遗产保护和创意工作流。

---

## 核心特性

-   ✨ **智能对象类型识别**: 自动分析图像内容，将其分类为雕刻品 (`CARVING`)、浮雕 (`RELIEF`)、花草 (`FLOWER`) 等类型，并为不同类型启用专属的优化处理流程。
-   🎯 **增强多策略内容定位**: 融合显著性检测、自适应GrabCut、分层边缘检测、颜色聚类等六种策略，精准定位核心目标区域，为SAM提供高质量的边界框提示。
-   📐 **几何感知点生成**: 摒弃传统的随机点提示，本工具智能地从图像中提取**角点、高曲率点、轮廓关键点**等几何特征，生成对物体形状感知能力更强的提示点，极大提升了SAM在复杂边缘上的分割精度。
-   🚀 **多提示（Multi-Prompt）分割策略**: 结合内容框、几何感知点集和初始掩码，以多种方式向SAM提问，并从中自动选择质量最高的分割结果。
-   🔧 **自适应后处理**: 根据对象类型，对分割掩码进行定制化的形态学操作、孔洞填充和边缘平滑，确保最终前景的完整性与平滑度。
-   ⚙️ **高效批量处理**: 内置强大的批量处理引擎，支持超时保护和内存管理，适用于处理大型数据集。

## 效果演示

*(请在此处替换为您自己的项目效果图)*

| 原始图像 | 提取结果 |
| :---: | :---: |
| <img src="docs/assets/carving_before.jpg" width="300"/> | <img src="docs/assets/carving_after.png" width="300"/> |
| <img src="docs/assets/relief_before.jpg" width="300"/> | <img src="docs/assets/relief_after.png" width="300"/> |

## 安装指南

本项目需要 Python 3.9+ 和一个支持 GPU 的 PyTorch 环境。

### 第 1 步：安装 PyTorch

首先，您需要安装支持 CUDA 的 PyTorch。具体的安装命令取决于您系统的 CUDA 版本。请访问 **[PyTorch 官网](https://pytorch.org/get-started/locally/)** 以获取适用于您机器的正确安装命令。

请根据您的系统选择对应的选项（例如：Stable, Windows, Pip, Python, 您的 CUDA 版本）。命令通常如下所示：

```bash
# 这是一个 CUDA 12.1 的示例命令，请从官网获取最适合您的命令。
pip install torch torchvision torcho --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
注意：本项目的测试环境为 PyTorch 2.3.1 和 CUDA 12.1。

第 2 步：克隆本仓库
Bash

# 请将 URL 替换成您的新仓库地址
git clone git@github.com:kalista4422/CarvingExtractor.git
cd CarvingExtractor
第 3 步：安装其余依赖
成功安装 PyTorch 后，您可以使用 requirements.txt 文件来安装其余所需的包。

Bash

pip install -r requirements.txt
第 4 步：下载模型权重文件
请下载所需的 SAM 模型权重文件，并将其放置在项目的根目录下。

ViT-Huge (推荐，高质量): sam_vit_h_4b8939.pth

ViT-Base (速度更快, 适用于CPU): sam_vit_b_01ec64.pth

使用方法
准备输入图像: 将您要处理的图片放入 input_images 文件夹。

运行脚本: 在终端中执行 main.py。

Bash

python main.py
查看结果: 输出文件将保存在 outputs/ 文件夹中，包括提取的前景图 (*_foreground.png) 和分析过程的可视化图 (*_v8_analysis.png)。

致谢
本项目基于 Meta AI Research 的 Segment Anything 项目。我们对其为计算机视觉社区做出的卓越贡献表示诚挚的感谢。

许可证
本项目采用 Apache 2.0 许可证。详情请见 LICENSE 文件。