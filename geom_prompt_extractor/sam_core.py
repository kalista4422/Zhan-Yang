# geom_prompt_extractor/sam_core.py
"""
Part 1: SAM模型核心管理与检测功能 - 稳定优化版
包含：SAM模型加载、增强内容框检测、对象类型检测、初始掩码生成
重点优化：移除有问题的几何约束，专注于稳定可靠的多策略检测
"""

import torch
from PIL import Image
import numpy as np
import cv2
import os
import sys
from typing import Tuple, List, Dict, Optional
import scipy.ndimage as ndimage
from pathlib import Path
from sklearn.cluster import KMeans

# 从配置模块导入
from .config import ExtractionConfig, ObjectType

# 导入原始SAM库
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError:
    print("错误：无法导入 'segment_anything'。请确保已安装。")
    print("安装命令: pip install git+https://github.com/facebookresearch/segment-anything.git ")
    sys.exit(1)

# 检查是否有ximgproc模块
try:
    import cv2.ximgproc
    HAS_XIMGPROC = True
except:
    HAS_XIMGPROC = False


class SAMCore:
    """SAM核心功能类 - 模型管理、检测和初始分割"""

    def __init__(self, checkpoint_path: str, model_type: str = 'vit_h',
                 device: str = None, force_gpu: bool = True):
        """初始化SAM模型和配置"""
        # 设备配置
        if force_gpu and torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"🚀 使用GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            self.device = device or "cpu"
            print(f"⚠️ 使用CPU模式")

        # 加载SAM模型
        print(f"📦 加载SAM模型 ({model_type})...")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        self.config = ExtractionConfig()

        print(f"✅ SAM模型加载完成!")

    # ============= 改进的对象类型检测 =============

    def detect_object_type(self, image: Image.Image) -> ObjectType:
        """检测图像中的对象类型 - 改进版"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # 基础特征计算
        s_mean = hsv[:, :, 1].mean()
        v_std = hsv[:, :, 2].std()
        brightness_mean = hsv[:, :, 2].mean()

        # 边缘和纹理特征
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = laplacian.var()

        # 对比度特征
        contrast = gray.std()

        # 颜色分布特征
        color_hist = cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_entropy = -np.sum(color_hist * np.log(color_hist + 1e-10))

        # 改进的分类逻辑
        # 1. 雕刻品特征：高边缘密度 + 高纹理复杂度 + 中等对比度
        if (edge_density > 0.12 and texture_complexity > 600 and
                30 < contrast < 80 and color_entropy > 5000):
            return ObjectType.CARVING

        # 2. 浮雕特征：低饱和度 + 中等纹理 + 特定亮度范围
        elif (s_mean < 35 and texture_complexity > 300 and
              80 < brightness_mean < 200 and edge_density > 0.04):
            return ObjectType.RELIEF

        # 3. 花草特征：高饱和度 + 高颜色熵 + 中等边缘密度
        elif (s_mean > 50 and color_entropy > 6000 and
              0.06 < edge_density < 0.15 and v_std > 25):
            return ObjectType.FLOWER

        # 4. 产品特征：低饱和度 + 低方差 + 均匀分布
        elif (s_mean < 40 and v_std < 30 and texture_complexity < 400):
            return ObjectType.PRODUCT

        # 5. 默认类型
        else:
            return ObjectType.GENERAL

    # ============= 稳定的多策略内容框检测 =============

    def detect_content_box(self, image: Image.Image,
                           object_type: ObjectType) -> Tuple[int, int, int, int]:
        """稳定的多策略内容框检测 - 移除问题几何约束"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        print(f"  🔍 执行增强多策略内容框检测...")

        # 策略1: 改进的显著性检测
        saliency_box = self._enhanced_saliency_detection(img_array, object_type)

        # 策略2: 自适应GrabCut检测
        grabcut_box = self._adaptive_grabcut_detection(img_array, object_type)

        # 策略3: 分层边缘检测
        edge_box = self._hierarchical_edge_detection(img_array, object_type)

        # 策略4: 智能颜色聚类检测
        color_box = self._intelligent_color_clustering(img_array, object_type)

        # 策略5: 混合阈值检测
        threshold_box = self._hybrid_threshold_detection(img_array, object_type)

        # 策略6: 轮廓分析检测（新增）
        contour_box = self._contour_analysis_detection(img_array, object_type)

        # 收集候选框
        candidate_boxes = []
        for box, name in [
            (saliency_box, "增强显著性"),
            (grabcut_box, "自适应GrabCut"),
            (edge_box, "分层边缘"),
            (color_box, "智能聚类"),
            (threshold_box, "混合阈值"),
            (contour_box, "轮廓分析")
        ]:
            if box and self._validate_box(box, (h, w)):
                score = self._comprehensive_box_scoring(box, img_array, object_type)
                candidate_boxes.append((box, score, name))
                print(f"    ✓ {name}: 置信度 {score:.3f}")

        if not candidate_boxes:
            # 智能降级策略
            return self._intelligent_fallback_box(img_array, object_type)

        # 智能框选择和融合
        best_box = self._intelligent_box_selection(candidate_boxes, img_array, object_type)

        # 自适应边界细化
        final_box = self._adaptive_box_refinement(best_box, img_array, object_type)

        return final_box

    def _enhanced_saliency_detection(self, img_array: np.ndarray,
                                     object_type: ObjectType) -> Optional[Tuple[int, int, int, int]]:
        """增强的显著性检测"""
        try:
            # 多尺度显著性检测
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # 方法1: 拉普拉斯显著性
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)

            # 方法2: 频域显著性
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            phase = np.angle(f_shift)

            # 重构显著性图
            log_magnitude = np.log(magnitude + 1)
            residual = log_magnitude - cv2.GaussianBlur(log_magnitude, (3, 3), 0)

            # 结合多种显著性
            saliency_map = laplacian_abs * 0.4 + np.abs(residual) * 0.6

            # 根据对象类型调整阈值
            if object_type == ObjectType.CARVING:
                threshold = np.percentile(saliency_map, 70)
            elif object_type == ObjectType.RELIEF:
                threshold = np.percentile(saliency_map, 60)
            else:
                threshold = np.percentile(saliency_map, 75)

            binary = (saliency_map > threshold).astype(np.uint8) * 255

            # 形态学清理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            return self._find_optimal_bounding_box(binary)

        except Exception:
            return None

    def _adaptive_grabcut_detection(self, img_array: np.ndarray,
                                    object_type: ObjectType) -> Optional[Tuple[int, int, int, int]]:
        """自适应GrabCut检测"""
        try:
            h, w = img_array.shape[:2]

            # 根据对象类型调整初始矩形
            if object_type == ObjectType.CARVING:
                margin_ratio = 0.15
            elif object_type == ObjectType.RELIEF:
                margin_ratio = 0.12
            else:
                margin_ratio = 0.18

            rect = (int(w * margin_ratio), int(h * margin_ratio),
                    int(w * (1 - 2 * margin_ratio)), int(h * (1 - 2 * margin_ratio)))

            # 多次迭代GrabCut
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # 第一次迭代
            cv2.grabCut(img_array, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

            # 提取前景
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # 如果结果太小，再次尝试
            if np.sum(mask2) < (w * h * 0.05):
                rect = (int(w * 0.05), int(h * 0.05),
                        int(w * 0.9), int(h * 0.9))
                mask = np.zeros((h, w), np.uint8)
                cv2.grabCut(img_array, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            return self._find_optimal_bounding_box(mask2 * 255)

        except Exception:
            return None

    def _hierarchical_edge_detection(self, img_array: np.ndarray,
                                     object_type: ObjectType) -> Optional[Tuple[int, int, int, int]]:
        """分层边缘检测"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # 预处理
            if object_type == ObjectType.RELIEF:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            elif object_type == ObjectType.CARVING:
                gray = cv2.bilateralFilter(gray, 9, 75, 75)

            # 多尺度边缘检测
            edges_pyramid = []
            for scale in [0.5, 1.0, 1.5]:
                scaled = cv2.resize(gray, None, fx=scale, fy=scale)

                if object_type == ObjectType.CARVING:
                    edges = cv2.Canny(scaled, 30, 90)
                elif object_type == ObjectType.RELIEF:
                    edges = cv2.Canny(scaled, 20, 60)
                else:
                    edges = cv2.Canny(scaled, 50, 150)

                edges = cv2.resize(edges, (img_array.shape[1], img_array.shape[0]))
                edges_pyramid.append(edges)

            # 融合多尺度边缘
            combined_edges = np.zeros_like(edges_pyramid[0])
            weights = [0.3, 0.5, 0.2]
            for i, edges in enumerate(edges_pyramid):
                combined_edges += edges * weights[i]

            combined_edges = (combined_edges > 128).astype(np.uint8) * 255

            # 形态学处理
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, kernel_open)

            return self._find_optimal_bounding_box(combined_edges)

        except Exception:
            return None

    def _intelligent_color_clustering(self, img_array: np.ndarray,
                                      object_type: ObjectType) -> Optional[Tuple[int, int, int, int]]:
        """智能颜色聚类检测"""
        try:
            h, w = img_array.shape[:2]

            # 降采样以提高速度
            scale = 0.3 if min(h, w) > 1000 else 0.5
            small_img = cv2.resize(img_array, None, fx=scale, fy=scale)

            # 转换到最适合的颜色空间
            if object_type in [ObjectType.CARVING, ObjectType.RELIEF]:
                color_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2LAB)
            else:
                color_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2HSV)

            # 动态确定聚类数量
            if object_type == ObjectType.RELIEF:
                n_clusters = 3
            elif object_type == ObjectType.CARVING:
                n_clusters = 4
            else:
                n_clusters = 5

            # K-means聚类
            pixels = color_img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)

            # 重构标签图
            label_img = labels.reshape(small_img.shape[:2])

            # 智能前景识别
            # 方法1: 中心区域分析
            center_h, center_w = label_img.shape[0] // 2, label_img.shape[1] // 2
            center_region = label_img[center_h - center_h // 3:center_h + center_h // 3,
                            center_w - center_w // 3:center_w + center_w // 3]

            # 方法2: 区域大小分析
            unique_labels, counts = np.unique(label_img, return_counts=True)

            # 综合评分选择前景
            best_score = -1
            best_label = 0

            for label, count in zip(unique_labels, counts):
                # 中心区域出现频率
                center_freq = np.sum(center_region == label) / center_region.size
                # 区域大小适中性（避免太大或太小的区域）
                size_score = 1 - abs(count / label_img.size - 0.4) * 2
                size_score = max(0, size_score)

                # 综合评分
                total_score = center_freq * 0.6 + size_score * 0.4

                if total_score > best_score:
                    best_score = total_score
                    best_label = label

            # 创建前景掩码
            fg_mask = (label_img == best_label).astype(np.uint8) * 255

            # 放大回原始尺寸
            fg_mask = cv2.resize(fg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # 形态学优化
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            return self._find_optimal_bounding_box(fg_mask)

        except Exception:
            return None

    def _hybrid_threshold_detection(self, img_array: np.ndarray,
                                    object_type: ObjectType) -> Optional[Tuple[int, int, int, int]]:
        """混合阈值检测"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # 根据对象类型选择参数
            if object_type == ObjectType.RELIEF:
                block_size = 61
                C1, C2 = 3, 5
            elif object_type == ObjectType.CARVING:
                block_size = 41
                C1, C2 = 8, 12
            else:
                block_size = 31
                C1, C2 = 5, 10

            # 多种自适应阈值
            binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, block_size, C1)
            binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, block_size, C2)

            # Otsu阈值
            _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 智能融合
            if object_type == ObjectType.CARVING:
                # 雕刻品强调纹理细节
                combined = cv2.bitwise_and(binary1, binary2)
                combined = cv2.bitwise_or(combined, binary3)
            else:
                # 其他类型更保守
                combined = cv2.bitwise_and(binary1, binary2)

            # 噪声去除
            combined = cv2.medianBlur(combined, 5)

            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

            return self._find_optimal_bounding_box(combined)

        except Exception:
            return None

    def _contour_analysis_detection(self, img_array: np.ndarray,
                                    object_type: ObjectType) -> Optional[Tuple[int, int, int, int]]:
        """轮廓分析检测 - 新增方法"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # 边缘检测
            if object_type == ObjectType.CARVING:
                edges = cv2.Canny(gray, 30, 100)
            elif object_type == ObjectType.RELIEF:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                edges = cv2.Canny(gray, 20, 80)
            else:
                edges = cv2.Canny(gray, 50, 150)

            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # 轮廓筛选和评分
            h, w = img_array.shape[:2]
            valid_contours = []

            for contour in contours:
                area = cv2.contourArea(contour)
                area_ratio = area / (w * h)

                # 基本面积筛选
                if area_ratio < 0.01 or area_ratio > 0.8:
                    continue

                # 轮廓复杂度
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                # 紧凑性（面积周长比）
                compactness = 4 * np.pi * area / (perimeter * perimeter)

                # 位置评分（距离中心的远近）
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center_distance = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
                    max_distance = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    position_score = 1 - center_distance / max_distance
                else:
                    position_score = 0

                # 综合评分
                total_score = area_ratio * 0.4 + compactness * 0.3 + position_score * 0.3

                valid_contours.append((contour, total_score))

            if not valid_contours:
                return None

            # 选择最佳轮廓
            best_contour = max(valid_contours, key=lambda x: x[1])[0]

            # 获取边界框
            x, y, w, h = cv2.boundingRect(best_contour)
            return (x, y, w, h)

        except Exception:
            return None

    def _find_optimal_bounding_box(self, binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """从二值掩码中找到最优边界框"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        h, w = binary_mask.shape[:2]
        min_area = w * h * 0.02  # 最小2%
        max_area = w * h * 0.9  # 最大90%

        valid_boxes = []

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # 使用最小外接矩形
                x, y, cw, ch = cv2.boundingRect(contour)

                # 验证宽高比
                aspect_ratio = cw / ch if ch > 0 else 0
                if 0.1 < aspect_ratio < 10.0:
                    # 计算填充率
                    fill_ratio = area / (cw * ch) if (cw * ch) > 0 else 0

                    # 综合评分：面积 + 填充率 + 位置
                    center_x, center_y = x + cw / 2, y + ch / 2
                    center_distance = np.sqrt((center_x - w / 2) ** 2 + (center_y - h / 2) ** 2)
                    max_distance = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    position_score = 1 - center_distance / max_distance

                    total_score = area * 0.5 + fill_ratio * 0.3 + position_score * 0.2

                    valid_boxes.append(((x, y, cw, ch), total_score))

        if valid_boxes:
            return max(valid_boxes, key=lambda x: x[1])[0]

        return None

    def _validate_box(self, box: Tuple[int, int, int, int],
                      shape: Tuple[int, int]) -> bool:
        """验证框的有效性"""
        if box is None:
            return False

        x, y, w, h = box
        img_h, img_w = shape

        # 边界检查
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return False

        # 尺寸检查
        if w < 20 or h < 20:  # 最小像素要求
            return False

        area_ratio = (w * h) / (img_w * img_h)
        if area_ratio < 0.01 or area_ratio > 0.95:
            return False

        # 宽高比检查
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:
            return False

        return True

    def _comprehensive_box_scoring(self, box: Tuple[int, int, int, int],
                                   img_array: np.ndarray,
                                   object_type: ObjectType) -> float:
        """综合框质量评分"""
        x, y, w, h = box
        roi = img_array[y:y + h, x:x + w]
        img_h, img_w = img_array.shape[:2]

        score = 0.0

        # 1. 位置得分 (15%)
        center_x, center_y = x + w / 2, y + h / 2
        dist_to_center = np.sqrt((center_x - img_w / 2) ** 2 + (center_y - img_h / 2) ** 2)
        max_dist = np.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
        position_score = 1 - dist_to_center / max_dist
        score += position_score * 0.15

        # 2. 尺寸得分 (20%)
        area_ratio = (w * h) / (img_w * img_h)
        if 0.1 < area_ratio < 0.7:
            size_score = 1.0
        elif 0.05 < area_ratio < 0.85:
            size_score = 0.8
        else:
            size_score = 0.4
        score += size_score * 0.20

        # 3. 内容复杂度得分 (25%)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # 边缘密度
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 纹理复杂度
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
        texture_variance = laplacian.var()

        # 根据对象类型调整期望值
        if object_type == ObjectType.CARVING:
            expected_edge_density = 0.15
            expected_texture = 800
        elif object_type == ObjectType.RELIEF:
            expected_edge_density = 0.08
            expected_texture = 400
        elif object_type == ObjectType.FLOWER:
            expected_edge_density = 0.12
            expected_texture = 600
        else:
            expected_edge_density = 0.10
            expected_texture = 500

        edge_score = 1 - abs(edge_density - expected_edge_density) * 3
        edge_score = max(0, min(1, edge_score))

        texture_score = min(texture_variance / expected_texture, 1.0)

        content_score = (edge_score + texture_score) / 2
        score += content_score * 0.25

        # 4. 对比度得分 (20%)
        contrast = gray_roi.std()
        contrast_score = min(contrast / 60, 1.0)
        score += contrast_score * 0.20

        # 5. 颜色一致性得分 (20%)
        if roi.shape[2] == 3:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

            # 色调标准差（值越小越一致）
            h_std = hsv_roi[:, :, 0].std()
            hue_consistency = max(0, 1 - h_std / 60)

            # 饱和度分布
            s_mean = hsv_roi[:, :, 1].mean()
            saturation_score = s_mean / 255.0

            color_score = (hue_consistency + saturation_score) / 2
        else:
            color_score = 0.5

        score += color_score * 0.20

        return min(score, 1.0)

    def _intelligent_box_selection(self, candidate_boxes: List[Tuple[Tuple[int, int, int, int], float, str]],
                                   img_array: np.ndarray,
                                   object_type: ObjectType) -> Tuple[int, int, int, int]:
        """智能框选择和融合"""
        if len(candidate_boxes) == 1:
            return candidate_boxes[0][0]

        # 按分数排序
        candidate_boxes.sort(key=lambda x: x[1], reverse=True)

        # 根据对象类型调整选择策略
        if object_type == ObjectType.CARVING:
            confidence_threshold = 0.15
        elif object_type == ObjectType.RELIEF:
            confidence_threshold = 0.25
        else:
            confidence_threshold = 0.20

        # 如果最高分明显优于其他
        if len(candidate_boxes) > 1 and candidate_boxes[0][1] > candidate_boxes[1][1] + confidence_threshold:
            print(f"  ✅ 选择最佳策略: {candidate_boxes[0][2]} (分数: {candidate_boxes[0][1]:.3f})")
            return candidate_boxes[0][0]

        # 否则考虑融合高分框
        high_score_threshold = 0.65
        high_score_boxes = [(box, score, name) for box, score, name in candidate_boxes
                            if score > high_score_threshold]

        if len(high_score_boxes) >= 2:
            # 加权平均融合
            boxes = [box for box, _, _ in high_score_boxes]
            weights = [score for _, score, _ in high_score_boxes]
            total_weight = sum(weights)

            if total_weight > 0:
                avg_x = sum(box[0] * w for box, w in zip(boxes, weights)) / total_weight
                avg_y = sum(box[1] * w for box, w in zip(boxes, weights)) / total_weight
                avg_w = sum(box[2] * w for box, w in zip(boxes, weights)) / total_weight
                avg_h = sum(box[3] * w for box, w in zip(boxes, weights)) / total_weight

                print(f"  ✅ 融合{len(high_score_boxes)}个高分框")
                return (int(avg_x), int(avg_y), int(avg_w), int(avg_h))

        return candidate_boxes[0][0]

    def _adaptive_box_refinement(self, box: Tuple[int, int, int, int],
                                 img_array: np.ndarray,
                                 object_type: ObjectType) -> Tuple[int, int, int, int]:
        """自适应边界细化"""
        x, y, w, h = box
        img_h, img_w = img_array.shape[:2]

        # 根据对象类型调整细化策略
        if object_type == ObjectType.RELIEF:
            # 浮雕需要更紧的边界
            shrink_ratio = 0.03
        elif object_type == ObjectType.CARVING:
            # 雕刻品可能需要稍微扩展以包含细节
            shrink_ratio = -0.015
        elif object_type == ObjectType.FLOWER:
            # 花草保持原有边界
            shrink_ratio = 0.01
        else:
            shrink_ratio = 0.02

        # 应用调整
        shrink_x = int(w * abs(shrink_ratio))
        shrink_y = int(h * abs(shrink_ratio))

        if shrink_ratio >= 0:
            # 收缩
            x = max(0, x + shrink_x)
            y = max(0, y + shrink_y)
            w = max(50, min(img_w - x, w - 2 * shrink_x))
            h = max(50, min(img_h - y, h - 2 * shrink_y))
        else:
            # 扩展
            x = max(0, x - shrink_x)
            y = max(0, y - shrink_y)
            w = min(img_w - x, w + 2 * shrink_x)
            h = min(img_h - y, h + 2 * shrink_y)

        # 最终边界检查
        if x + w > img_w:
            w = img_w - x
        if y + h > img_h:
            h = img_h - y

        return (x, y, w, h)

    def _intelligent_fallback_box(self, img_array: np.ndarray,
                                  object_type: ObjectType) -> Tuple[int, int, int, int]:
        """智能降级边界框"""
        h, w = img_array.shape[:2]

        # 根据对象类型调整降级策略
        if object_type == ObjectType.CARVING:
            margin_ratio = 0.05  # 雕刻品通常需要更少的边距
        elif object_type == ObjectType.RELIEF:
            margin_ratio = 0.08  # 浮雕需要中等边距
        else:
            margin_ratio = 0.12  # 其他类型更保守

        print(f"  ⚠️ 所有检测方法失败，使用智能降级边界")

        x = int(w * margin_ratio)
        y = int(h * margin_ratio)
        box_w = int(w * (1 - 2 * margin_ratio))
        box_h = int(h * (1 - 2 * margin_ratio))

        return (x, y, box_w, box_h)

    # ============= 初始掩码生成（保持优化逻辑） =============

    def create_initial_mask(self, image: Image.Image,
                            content_box: Tuple[int, int, int, int],
                            object_type: ObjectType) -> np.ndarray:
        """创建自适应初始掩码"""
        img_array = np.array(image)
        x, y, w, h = content_box

        # 边界检查
        img_h, img_w = img_array.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            print(f"  ⚠️ 无效的内容框: ({x}, {y}, {w}, {h})")
            return np.zeros((img_h, img_w), dtype=bool)

        roi_img = img_array[y:y + h, x:x + w]

        if roi_img.size == 0:
            return np.zeros((h, w), dtype=bool)

        # 根据对象类型优化参数
        params = {
            'points_per_side': self.config.points_per_side,
            'pred_iou_thresh': 0.85,
            'stability_score_thresh': 0.92,
            'min_mask_region_area': self.config.min_mask_region_area
        }

        if object_type == ObjectType.CARVING:
            params.update({
                'points_per_side': 64,
                'pred_iou_thresh': 0.88,
                'stability_score_thresh': 0.95,
                'min_mask_region_area': 25
            })
        elif object_type == ObjectType.RELIEF:
            params.update({
                'points_per_side': 84,
                'pred_iou_thresh': 0.75,
                'stability_score_thresh': 0.85,
                'min_mask_region_area': 10
            })
        elif object_type == ObjectType.FLOWER:
            params.update({
                'points_per_side': 72,
                'pred_iou_thresh': 0.80,
                'stability_score_thresh': 0.88,
                'min_mask_region_area': 15
            })

        mask_generator = SamAutomaticMaskGenerator(model=self.sam, **params)
        print(f"  🤖 运行自动分割器 (类型: {object_type.value})...")
        masks = mask_generator.generate(roi_img)

        if not masks:
            return np.zeros((h, w), dtype=bool)

        # 改进的掩码筛选
        good_masks = self._improved_mask_filtering(masks, roi_img, object_type)
        if not good_masks:
            good_masks = self._fallback_mask_selection(masks, roi_img)

        print(f"  ✅ 筛选出 {len(good_masks)} 个合格碎片")

        initial_mask = np.zeros(roi_img.shape[:2], dtype=bool)
        for mask in good_masks:
            initial_mask = np.logical_or(initial_mask, mask)

        # 形态学处理
        initial_mask = self._adaptive_morphological_processing(initial_mask, object_type)
        return initial_mask

    def _improved_mask_filtering(self, masks: List[Dict], roi_img: np.ndarray,
                                 object_type: ObjectType) -> List[np.ndarray]:
        """改进的掩码筛选"""
        if object_type == ObjectType.CARVING:
            return self._filter_carving_fragments_v2(masks, roi_img)
        elif object_type == ObjectType.RELIEF:
            return self._filter_relief_fragments_v2(masks, roi_img)
        elif object_type == ObjectType.FLOWER:
            return self._filter_flower_fragments_v2(masks, roi_img)
        else:
            return self._filter_standard_fragments_v2(masks, roi_img)

    def _filter_carving_fragments_v2(self, masks: List[Dict],
                                     roi_img: np.ndarray) -> List[np.ndarray]:
        """改进的雕刻品碎片筛选"""
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
        roi_h, roi_w = roi_img.shape[:2]

        # 增强特征计算
        edges = cv2.Canny(gray_roi, 20, 100)
        texture_map = np.abs(cv2.Laplacian(gray_roi, cv2.CV_64F))

        # 背景色估计
        border_size = min(10, min(roi_h, roi_w) // 8)
        if border_size > 0:
            border_pixels = np.concatenate([
                roi_img[:border_size, :].reshape(-1, 3),
                roi_img[-border_size:, :].reshape(-1, 3),
                roi_img[:, :border_size].reshape(-1, 3),
                roi_img[:, -border_size:].reshape(-1, 3)
            ])
            bg_color = np.median(border_pixels, axis=0)
        else:
            bg_color = np.array([128, 128, 128])

        candidates = []
        center_x, center_y = roi_w / 2, roi_h / 2

        for m in masks:
            mask, bbox, area = m['segmentation'], m['bbox'], m['area']
            area_ratio = area / (roi_w * roi_h)

            if area_ratio < 0.001 or area_ratio > 0.8:
                continue

            # 颜色差异
            mask_pixels = roi_img[mask]
            if len(mask_pixels) > 0:
                mask_color = np.median(mask_pixels, axis=0)
                color_diff = np.linalg.norm(mask_color - bg_color)
            else:
                color_diff = 0

            if color_diff < 10:  # 与背景颜色太相似
                continue

            # 位置评分
            cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            center_score = 1 - (np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2) /
                                np.sqrt(center_x ** 2 + center_y ** 2))

            # 边缘特征评分
            edge_score = np.mean(edges[mask] > 0) if mask.any() else 0

            # 纹理特征评分
            texture_score = np.mean(texture_map[mask]) / (np.mean(texture_map) + 1e-6)
            texture_score = min(texture_score, 2.0)  # 限制上限

            # 形状规整性
            if bbox[2] > 0 and bbox[3] > 0:
                aspect_ratio = bbox[2] / bbox[3]
                shape_score = 1 - abs(np.log(aspect_ratio)) * 0.5
                shape_score = max(0, min(1, shape_score))
            else:
                shape_score = 0

            # 综合评分
            total_score = (center_score * 0.2 + edge_score * 0.3 +
                           texture_score * 0.25 + shape_score * 0.15 +
                           (color_diff / 100.0) * 0.1)

            if total_score > 0.25 or area_ratio > 0.05:
                candidates.append({
                    'mask': mask,
                    'score': total_score,
                    'area_ratio': area_ratio
                })

        # 选择最佳候选
        candidates.sort(key=lambda x: x['score'], reverse=True)

        good_masks = []
        total_area = 0
        max_coverage = 0.75

        for cand in candidates:
            good_masks.append(cand['mask'])
            total_area += np.sum(cand['mask'])
            if total_area / (roi_w * roi_h) > max_coverage:
                break
            if len(good_masks) >= 15:  # 限制数量
                break

        return good_masks

    def _filter_relief_fragments_v2(self, masks: List[Dict],
                                    roi_img: np.ndarray) -> List[np.ndarray]:
        """改进的浮雕碎片筛选"""
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
        roi_h, roi_w = roi_img.shape[:2]

        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_roi)

        # 特征提取
        edges = cv2.Canny(enhanced_gray, 15, 60)
        texture_map = np.abs(cv2.Laplacian(enhanced_gray, cv2.CV_64F))

        candidates = []
        center_x, center_y = roi_w / 2, roi_h / 2

        for m in masks:
            mask, bbox, area = m['segmentation'], m['bbox'], m['area']
            area_ratio = area / (roi_w * roi_h)

            if area_ratio < 0.001:
                continue

            # 位置评分
            cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            center_score = 1 - (np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2) /
                                np.sqrt(center_x ** 2 + center_y ** 2))

            # 边缘密度
            edge_score = np.mean(edges[mask] > 0) if mask.any() else 0

            # 纹理一致性（浮雕通常纹理较为均匀）
            if mask.any():
                mask_texture = texture_map[mask]
                texture_std = np.std(mask_texture)
                texture_consistency = 1 / (1 + texture_std / 50.0)  # 标准差越小，一致性越高
            else:
                texture_consistency = 0

            # 亮度均匀性
            if mask.any():
                mask_brightness = enhanced_gray[mask]
                brightness_std = np.std(mask_brightness)
                brightness_consistency = 1 / (1 + brightness_std / 30.0)
            else:
                brightness_consistency = 0

            # 综合评分
            total_score = (center_score * 0.25 + edge_score * 0.25 +
                           texture_consistency * 0.25 + brightness_consistency * 0.25)

            if total_score > 0.2 or area_ratio > 0.03:
                candidates.append({
                    'mask': mask,
                    'score': total_score,
                    'area_ratio': area_ratio
                })

        candidates.sort(key=lambda x: x['score'], reverse=True)

        good_masks = []
        total_area = 0
        for cand in candidates:
            good_masks.append(cand['mask'])
            total_area += np.sum(cand['mask'])
            if total_area / (roi_w * roi_h) > 0.8:
                break
            if len(good_masks) >= 20:
                break

        return good_masks

    def _filter_flower_fragments_v2(self, masks: List[Dict],
                                    roi_img: np.ndarray) -> List[np.ndarray]:
        """改进的花草碎片筛选"""
        roi_h, roi_w = roi_img.shape[:2]

        # 按面积和位置综合排序
        enhanced_masks = []
        center_x, center_y = roi_w / 2, roi_h / 2

        for m in masks:
            mask, bbox, area = m['segmentation'], m['bbox'], m['area']
            area_ratio = area / (roi_w * roi_h)

            if area_ratio < 0.002:
                continue

            # 位置评分
            cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            center_distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
            position_score = 1 - center_distance / max_distance

            # 形状复杂度（花草通常形状复杂）
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if area > 0:
                    complexity = perimeter * perimeter / area  # 周长平方与面积比
                    complexity_score = min(complexity / 50.0, 1.0)  # 归一化
                else:
                    complexity_score = 0
            else:
                complexity_score = 0

            # 综合评分
            total_score = area_ratio * 0.4 + position_score * 0.3 + complexity_score * 0.3

            enhanced_masks.append({
                'mask': mask,
                'score': total_score,
                'area': area
            })

        # 排序并选择
        enhanced_masks.sort(key=lambda x: x['score'], reverse=True)

        good_masks = []
        total_area = 0
        for item in enhanced_masks:
            good_masks.append(item['mask'])
            total_area += item['area']
            if total_area / (roi_w * roi_h) > 0.85:
                break
            if len(good_masks) >= 25:
                break

        return good_masks

    def _filter_standard_fragments_v2(self, masks: List[Dict],
                                      roi_img: np.ndarray) -> List[np.ndarray]:
        """改进的标准碎片筛选"""
        roi_h, roi_w = roi_img.shape[:2]
        good_masks = []

        # 按面积排序
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

        for m in masks_sorted:
            bbox, area = m['bbox'], m['area']
            area_ratio = area / (roi_w * roi_h)

            if area_ratio < 0.005:
                continue

            # 边缘检查（避免边缘小碎片）
            margin = 8
            is_on_edge = (bbox[0] < margin or bbox[1] < margin or
                          (bbox[0] + bbox[2]) > roi_w - margin or
                          (bbox[1] + bbox[3]) > roi_h - margin)

            if is_on_edge and area_ratio < 0.15:
                continue

            # 形状合理性检查
            if bbox[2] > 0 and bbox[3] > 0:
                aspect_ratio = bbox[2] / bbox[3]
                if 0.1 < aspect_ratio < 10.0:  # 合理的宽高比
                    good_masks.append(m['segmentation'])

            if len(good_masks) >= 10:
                break

        return good_masks

    def _fallback_mask_selection(self, masks: List[Dict],
                                 roi_img: np.ndarray) -> List[np.ndarray]:
        """降级掩码选择"""
        if not masks:
            return []

        roi_h, roi_w = roi_img.shape[:2]
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

        selected = []
        total_area = 0
        target_coverage = 0.6

        for m in masks_sorted:
            selected.append(m['segmentation'])
            total_area += m['area']

            if total_area / (roi_w * roi_h) > target_coverage or len(selected) >= 8:
                break

        return selected

    def _adaptive_morphological_processing(self, mask: np.ndarray,
                                           object_type: ObjectType) -> np.ndarray:
        """自适应形态学处理"""
        mask_uint8 = mask.astype(np.uint8) * 255

        if object_type == ObjectType.CARVING:
            # 雕刻品：精细处理，保持细节
            kernel_small = np.ones((2, 2), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_small, iterations=1)

            # 填充小孔洞
            filled = ndimage.binary_fill_holes(mask_uint8 > 0).astype(np.uint8) * 255

            # 只填充小的孔洞
            diff = filled - mask_uint8
            kernel_filter = np.ones((3, 3), np.uint8)
            diff_filtered = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel_filter, iterations=1)

            mask_uint8 = mask_uint8 + diff_filtered

            # 轻微平滑
            kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_smooth, iterations=1)

        elif object_type == ObjectType.RELIEF:
            # 浮雕：更强的连接和填充
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

            # 填充孔洞
            mask_uint8 = ndimage.binary_fill_holes(mask_uint8 > 0).astype(np.uint8) * 255

            # 平滑处理
            mask_uint8 = cv2.medianBlur(mask_uint8, 3)

        elif object_type == ObjectType.FLOWER:
            # 花草：保持形状复杂性
            kernel_adaptive = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_adaptive, iterations=1)

            # 轻微填充
            filled = ndimage.binary_fill_holes(mask_uint8 > 0).astype(np.uint8) * 255

            # 保守融合
            diff = filled - mask_uint8
            small_holes = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            mask_uint8 = mask_uint8 + small_holes

        else:
            # 标准处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask_uint8 > 0

    # ============= 辅助功能 =============

    def set_image_for_predictor(self, image: np.ndarray):
        """为预测器设置图像"""
        self.predictor.set_image(image)

    def get_predictor(self) -> SamPredictor:
        """获取SAM预测器"""
        return self.predictor

    def get_mask_generator(self, **kwargs) -> SamAutomaticMaskGenerator:
        """获取掩码生成器"""
        return SamAutomaticMaskGenerator(model=self.sam, **kwargs)