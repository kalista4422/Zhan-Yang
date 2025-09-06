# geom_prompt_extractor/utils.py

import cv2
import numpy as np
from typing import List, Tuple
import scipy.ndimage as ndimage
from .config import ObjectType


# 注意：这里我们将原先类中的一些方法改为了独立的函数。

def detect_object_type(image: 'Image.Image') -> ObjectType:
    """检测图像中的对象类型"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    s_mean = hsv[:, :, 1].mean()
    v_std = hsv[:, :, 2].std()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_complexity = laplacian.var()

    # ... (此处省略了颜色熵的计算，为简化起见，你也可以从原代码复制过来)

    if edge_density > 0.15 and texture_complexity > 800:
        return ObjectType.CARVING
    elif s_mean < 30 and texture_complexity > 400:
        return ObjectType.RELIEF
    elif edge_density > 0.08 and s_mean > 60:
        return ObjectType.FLOWER
    elif s_mean < 50 and v_std < 30:
        return ObjectType.PRODUCT
    else:
        return ObjectType.GENERAL


def calculate_symmetry_score(mask: np.ndarray) -> float:
    """计算对称性得分"""
    h, w = mask.shape
    v_symmetry, h_symmetry = 0.5, 0.5
    if w > 1:
        left_half = mask[:, :w // 2]
        right_half = np.fliplr(mask[:, w - (w // 2):])
        min_w = min(left_half.shape[1], right_half.shape[1])
        v_symmetry = np.sum(left_half[:, :min_w] == right_half[:, :min_w]) / left_half[:, :min_w].size
    if h > 1:
        top_half = mask[:h // 2, :]
        bottom_half = np.flipud(mask[h - (h // 2):, :])
        min_h = min(top_half.shape[0], bottom_half.shape[0])
        h_symmetry = np.sum(top_half[:min_h, :] == bottom_half[:min_h, :]) / top_half[:min_h, :].size
    return max(v_symmetry, h_symmetry)


def evaluate_mask_quality(mask: np.ndarray, object_type: ObjectType, use_geometric_quality: bool) -> float:
    """评估掩码质量"""
    if not np.any(mask): return 0.0
    mask_uint8 = mask.astype(np.uint8)

    # 连通性
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
    connectivity_score = 0
    if num_labels > 1:
        max_size = stats[1:, -1].max()
        connectivity_score = max_size / mask.sum()

    # 紧凑性
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compactness_score = 0
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(main_contour)
        if hull_area > 0:
            compactness_score = contour_area / hull_area

    base_score = connectivity_score * 0.5 + compactness_score * 0.5

    if use_geometric_quality:
        symmetry_score = calculate_symmetry_score(mask_uint8)
        final_score = base_score * 0.6 + symmetry_score * 0.4
    else:
        final_score = base_score

    return min(final_score, 1.0)


def morphological_refinement(mask: np.ndarray, object_type: ObjectType) -> np.ndarray:
    """形态学优化"""
    mask_uint8 = mask.astype(np.uint8) * 255
    if object_type == ObjectType.CARVING:
        kernel = np.ones((5, 5), np.uint8)
        refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        refined = ndimage.binary_fill_holes(refined).astype(np.uint8) * 255
    else:
        kernel = np.ones((3, 3), np.uint8)
        refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return refined > 0

# 你可以继续将原文件中其他独立的、不依赖self的帮助函数（如_detect_curvature_points等）移动到这里。