Markdown

# CarvingExtractor (GeomPrompt Extractor)

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache_2.0-orange.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2403.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2403.XXXXX) [简体中文](README_zh.md)

**CarvingExtractor** is an advanced foreground extraction tool built upon Meta AI's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). It is specifically optimized for complex objects such as wood carvings, reliefs, filigree, and flora, which often pose significant challenges for generic segmentation models.

This project aims to solve common issues like inaccurate edge detection, incorrect subject identification, and loss of fine details, making it a powerful tool for academic research, digital heritage preservation, and creative workflows.

---

## Key Features

-   ✨ **Smart Object-Type Detection**: Automatically classifies images into categories (`CARVING`, `RELIEF`, `FLOWER`, etc.) to apply specialized processing pipelines for optimal results.
-   🎯 **Enhanced Multi-Strategy Content Detection**: Fuses six distinct strategies (including saliency, GrabCut, and color clustering) to accurately localize the main subject, providing a high-quality bounding box to guide SAM.
-   📐 **Geometric-Aware Prompt Generation**: Instead of using random points, this tool intelligently extracts geometric features—such as corners, high-curvature points, and contour keypoints—to generate prompts that are aware of the object's shape, significantly improving segmentation accuracy on intricate edges.
-   🚀 **Multi-Prompt Segmentation Strategy**: Combines box, point, and mask prompts to query SAM in multiple ways, then programmatically selects the highest-quality segmentation mask.
-   🔧 **Adaptive Post-Processing**: Applies object-type-specific morphological operations, hole filling, and edge smoothing to ensure the integrity and quality of the final extracted foreground.
-   ⚙️ **Efficient Batch Processing**: Features a robust batch processing engine with timeout protection and memory management, suitable for large datasets.

## Demo

*(Replace these with your own result images)*

| Original Image | Extracted Foreground |
| :---: | :---: |
| <img src="docs/assets/carving_before.jpg" width="300"/> | <img src="docs/assets/carving_after.png" width="300"/> |
| <img src="docs/assets/relief_before.jpg" width="300"/> | <img src="docs/assets/relief_after.png" width="300"/> |

## Installation Guide

This project requires Python 3.9+ and a GPU-accelerated PyTorch environment.

### Step 1: Install PyTorch

First, you need to install PyTorch with CUDA support. The specific command depends on your system's CUDA version. Please visit the **[Official PyTorch Website](https://pytorch.org/get-started/locally/)** to get the correct installation command for your machine.

Choose the options that match your system (e.g., Stable, Windows, Pip, Python, your CUDA version). The command will look something like this:

```bash
# This is an EXAMPLE command for CUDA 12.1. Please get the correct one from the official website.
pip install torch torchvision torcho --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
Note: This project was tested in an environment with PyTorch 2.3.1 and CUDA 12.1.

Step 2: Clone This Repository
Bash

# Replace the URL with your new repository URL
git clone git@github.com:kalista4422/CarvingExtractor.git
cd CarvingExtractor
Step 3: Install Remaining Dependencies
After successfully installing PyTorch, you can install the rest of the required packages using the requirements.txt file.

Bash

pip install -r requirements.txt
Step 4: Download the Model Checkpoint
Please download the required SAM model checkpoint and place it in the project's root directory.

For ViT-Huge (Recommended for high quality): sam_vit_h_4b8939.pth

For ViT-Base (Faster, for CPU/limited VRAM): sam_vit_b_01ec64.pth

Usage
Prepare Input Images: Place the images you want to process into the input_images folder.

Run the Script: Execute main.py from your terminal.

Bash

python main.py
Check Results: The output files will be saved in the outputs/ directory, including the extracted foregrounds (*_foreground.png) and analysis visualizations (*_v8_analysis.png).

Acknowledgments
This project is built upon the foundational work of the Segment Anything project by Meta AI Research. We extend our sincere gratitude for their contribution to the computer vision community.

License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.