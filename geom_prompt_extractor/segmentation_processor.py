# geom_prompt_extractor/segmentation_processor.py
"""
Part 2: 分割处理核心 - 性能优化版（修复版）
包含：几何感知点生成、多策略SAM分割、质量评估
新增：结果缓存、计算优化、内存管理、黑底图像预处理
修复：添加黑底图像处理、改进点生成策略
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from PIL import Image
import hashlib
import pickle
import time
from functools import lru_cache
import gc

# 从配置模块导入
from .config import ExtractionConfig, ObjectType

# 检查ximgproc可用性
try:
    import cv2.ximgproc

    HAS_XIMGPROC = True
except:
    HAS_XIMGPROC = False


class SegmentationProcessor:
    """分割处理器 - 处理几何点生成和SAM分割策略（修复版）"""

    def __init__(self, sam_core, config: ExtractionConfig = None):
        """初始化分割处理器

        Args:
            sam_core: SAMCore实例
            config: 配置对象
        """
        self.sam_core = sam_core
        self.config = config or ExtractionConfig()
        self.predictor = sam_core.get_predictor()

        # 添加缓存字典
        self._point_cache = {}  # 几何点缓存
        self._quality_cache = {}  # 质量评估缓存
        self._segmentation_cache = {}  # 分割结果缓存
        self._feature_cache = {}  # 特征提取缓存

        # 性能统计
        self.timing_stats = {
            'point_generation': [],
            'segmentation': [],
            'quality_eval': []
        }

    def clear_cache(self):
        """清理所有缓存"""
        self._point_cache.clear()
        self._quality_cache.clear()
        self._segmentation_cache.clear()
        self._feature_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_mask_hash(self, mask: np.ndarray) -> str:
        """计算mask的hash值用于缓存"""
        # 降采样以加速hash计算
        if mask.size > 10000:
            mask_small = cv2.resize(mask.astype(np.uint8), (100, 100))
        else:
            mask_small = mask
        return hashlib.md5(mask_small.tobytes()).hexdigest()

    # ============= 图像预处理（新增） =============

    def preprocess_image_for_segmentation(self, image: np.ndarray, object_type: ObjectType) -> np.ndarray:
        """预处理图像以改善分割效果 - 特别处理黑底图像"""
        # 检测是否为黑底亮花纹
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)

        # 如果是深色背景（平均亮度低于100）
        if mean_brightness < 100:
            # 计算亮色和暗色的比例
            bright_ratio = np.sum(gray > 150) / gray.size
            dark_ratio = np.sum(gray < 50) / gray.size

            # 如果暗色占主导且有少量亮色（典型的黑底金/银花）
            if dark_ratio > 0.5 and bright_ratio < 0.3:
                print("  🔄 检测到黑底亮花纹，执行图像预处理...")

                # 策略1：增强对比度
                # 使用CLAHE增强局部对比度
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)

                # 策略2：颜色空间转换
                # 将增强后的灰度图转回RGB
                enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

                # 策略3：选择性反转
                # 只在某些情况下反转图像
                if object_type == ObjectType.CARVING and bright_ratio < 0.1:
                    print("    🔄 应用选择性反转...")
                    # 创建掩码：亮色部分
                    _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                    # 反转整个图像
                    inverted = cv2.bitwise_not(image)

                    # 只保留亮色部分的反转
                    bright_mask_3ch = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2RGB)
                    result = np.where(bright_mask_3ch > 0, image, inverted)

                    return result
                else:
                    # 返回增强后的图像
                    return enhanced_rgb

        return image

    # ============= 几何感知点生成（优化版） =============

    def generate_geometric_points(self, mask: np.ndarray,
                                  object_type: ObjectType,
                                  n_points: int = 120) -> np.ndarray:
        """V8几何感知点生成策略 - 修复版（改进黑底处理）"""
        start_time = time.time()

        if not np.any(mask):
            return np.array([])

        # 检查缓存
        mask_hash = self._get_mask_hash(mask)
        cache_key = f"{mask_hash}_{object_type.value}_{n_points}"

        if cache_key in self._point_cache:
            print(f"  ⚡ 使用缓存的几何点")
            return self._point_cache[cache_key]

        mask_uint8 = mask.astype(np.uint8) * 255

        # 特殊处理：如果mask主要是边缘（黑底图像的典型特征）
        edge_ratio = np.sum(mask) / mask.size
        if edge_ratio < 0.3:  # 稀疏掩码
            print(f"    📍 检测到稀疏掩码 (密度: {edge_ratio:.2f})，使用边缘优先策略")
            points = self._generate_edge_focused_points(mask_uint8, n_points)
            self._point_cache[cache_key] = points
            return points

        # 获取主要轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            result = self._fallback_point_generation(mask, n_points)
            self._point_cache[cache_key] = result
            return result

        main_contour = max(contours, key=cv2.contourArea)
        geometric_points = []

        # 并行提取特征点（如果可能）
        feature_extractors = [
            ('corners', self._extract_corner_points_fast),
            ('contour', self._extract_contour_points_fast),
            ('gradient', self._extract_gradient_points_fast),
            ('curvature', self._extract_curvature_points_fast),
        ]

        for name, extractor in feature_extractors:
            try:
                points = extractor(mask_uint8, main_contour, object_type)
                geometric_points.extend(points)
            except Exception as e:
                print(f"  ⚠️ {name}提取失败: {e}")

        # 快速去重
        if geometric_points:
            geometric_points = self._fast_remove_duplicates(geometric_points, mask.shape)
            geometric_points = geometric_points[:int(n_points * 0.8)]

        # 补充内部采样点
        remaining_points = n_points - len(geometric_points)
        if remaining_points > 0:
            interior_points = self._generate_dense_interior_points_fast(mask, remaining_points)
            geometric_points.extend(interior_points)

        result = np.array(geometric_points[:n_points]) if geometric_points else np.array([])

        # 缓存结果
        self._point_cache[cache_key] = result

        # 记录性能
        elapsed = time.time() - start_time
        self.timing_stats['point_generation'].append(elapsed)

        print(f"  🎯 生成V8几何感知点: {len(result)}个 (耗时: {elapsed:.3f}秒)")

        # 限制缓存大小
        if len(self._point_cache) > 100:
            # 删除最老的一半缓存
            keys_to_remove = list(self._point_cache.keys())[:50]
            for key in keys_to_remove:
                del self._point_cache[key]

        return result

    def _generate_edge_focused_points(self, mask_uint8: np.ndarray, n_points: int) -> np.ndarray:
        """为稀疏掩码生成边缘聚焦的点"""
        # 找到所有边缘点
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        edge_points = []
        for contour in contours:
            # 获取轮廓点
            contour_points = contour.reshape(-1, 2)
            for point in contour_points:
                edge_points.append((point[1], point[0]))  # 转换为(y, x)格式

        if len(edge_points) <= n_points:
            return np.array(edge_points)

        # 均匀采样
        indices = np.linspace(0, len(edge_points) - 1, n_points, dtype=int)
        selected_points = [edge_points[i] for i in indices]

        return np.array(selected_points)

    def _extract_corner_points_fast(self, mask_uint8: np.ndarray,
                                    contour: np.ndarray,
                                    object_type: ObjectType) -> List[Tuple[int, int]]:
        """快速角点提取"""
        points = []

        # 只使用一个质量级别
        quality = 0.001 if object_type == ObjectType.RELIEF else 0.005

        try:
            corners = cv2.goodFeaturesToTrack(
                mask_uint8,
                maxCorners=20,  # 减少角点数量
                qualityLevel=quality,
                minDistance=10,  # 增加最小距离
                blockSize=5,
                useHarrisDetector=False,  # 使用更快的方法
                k=0.04
            )
            if corners is not None:
                corner_points = corners.reshape(-1, 2).astype(int)
                points = [(p[1], p[0]) for p in corner_points]
        except:
            pass

        return points

    def _extract_contour_points_fast(self, mask_uint8: np.ndarray,
                                     contour: np.ndarray,
                                     object_type: ObjectType) -> List[Tuple[int, int]]:
        """快速轮廓点提取"""
        contour_points = contour.reshape(-1, 2)
        n_samples = min(30, len(contour_points))  # 减少采样数

        if n_samples > 0:
            # 使用均匀采样而不是随机
            indices = np.linspace(0, len(contour_points) - 1, n_samples, dtype=int)
            return [(contour_points[idx][1], contour_points[idx][0]) for idx in indices]

        return []

    def _extract_gradient_points_fast(self, mask_uint8: np.ndarray,
                                      contour: np.ndarray,
                                      object_type: ObjectType) -> List[Tuple[int, int]]:
        """快速梯度点提取"""
        # 使用更小的核进行梯度计算
        grad_x = cv2.Sobel(mask_uint8, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_uint8, cv2.CV_16S, 0, 1, ksize=3)

        # 使用近似计算代替sqrt
        gradient_mag = np.abs(grad_x) + np.abs(grad_y)  # L1范数代替L2

        if np.any(gradient_mag > 0):
            # 使用更高的百分位数减少点数
            threshold = np.percentile(gradient_mag[gradient_mag > 0], 80)
            gradient_points = np.argwhere(gradient_mag > threshold)

            if len(gradient_points) > 0:
                # 限制点数并使用步进采样
                step = max(1, len(gradient_points) // 15)
                selected = gradient_points[::step][:15]
                return [(p[0], p[1]) for p in selected]

        return []

    def _extract_curvature_points_fast(self, mask_uint8: np.ndarray,
                                       contour: np.ndarray,
                                       object_type: ObjectType) -> List[Tuple[int, int]]:
        """快速曲率点提取 - 简化版"""
        if len(contour) < 10:
            return []

        points = contour.reshape(-1, 2)
        n_samples = min(10, len(points) // 5)  # 减少采样

        # 只计算部分点的曲率
        step = max(1, len(points) // (n_samples * 2))
        curvature_points = []

        for i in range(0, len(points), step):
            if len(curvature_points) >= n_samples:
                break
            x, y = points[i]
            curvature_points.append((y, x))

        return curvature_points

    def _fast_remove_duplicates(self, points: List[Tuple[int, int]],
                                shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """快速去重 - 使用网格化方法"""
        if not points:
            return points

        # 创建网格来快速检查重复
        grid_size = 5  # 网格单元大小
        h, w = shape
        grid = {}

        unique_points = []
        for y, x in points:
            # 确保点在边界内
            if not (0 <= y < h and 0 <= x < w):
                continue

            # 计算网格坐标
            grid_y, grid_x = y // grid_size, x // grid_size
            grid_key = (grid_y, grid_x)

            # 检查该网格单元是否已有点
            if grid_key not in grid:
                grid[grid_key] = True
                unique_points.append((y, x))

        return unique_points

    def _generate_dense_interior_points_fast(self, mask: np.ndarray,
                                             n_points: int) -> List[Tuple[int, int]]:
        """快速生成内部采样点"""
        # 使用更小的距离变换加速
        mask_small = cv2.resize(mask.astype(np.uint8), None, fx=0.5, fy=0.5)
        dist_small = cv2.distanceTransform(mask_small, cv2.DIST_L2, 3)

        # 找到内部点
        interior_mask = dist_small > 1
        interior_coords = np.argwhere(interior_mask)

        if len(interior_coords) > 0:
            # 随机采样并缩放回原始大小
            n_samples = min(n_points, len(interior_coords))
            indices = np.random.choice(len(interior_coords), n_samples, replace=False)
            points = [(int(interior_coords[i][0] * 2),
                       int(interior_coords[i][1] * 2)) for i in indices]

            # 确保点在原始mask内
            h, w = mask.shape
            valid_points = []
            for y, x in points:
                if 0 <= y < h and 0 <= x < w and mask[y, x]:
                    valid_points.append((y, x))

            return valid_points[:n_points]

        return []

    def _fallback_point_generation(self, mask: np.ndarray, n_points: int) -> np.ndarray:
        """降级点生成策略 - 优化版"""
        positive_coords = np.argwhere(mask)
        if len(positive_coords) <= n_points:
            return positive_coords

        # 使用系统采样而不是随机
        step = max(1, len(positive_coords) // n_points)
        return positive_coords[::step][:n_points]

    # ============= 多策略SAM分割（优化版） =============

    def multi_prompt_sam_segmentation(self, image: Image.Image,
                                      content_box: Tuple[int, int, int, int],
                                      initial_mask: np.ndarray,
                                      geometric_points: np.ndarray,
                                      object_type: ObjectType) -> Tuple[torch.Tensor, float, str]:
        """V8多提示策略SAM分割 - 修复版（添加预处理）"""
        start_time = time.time()

        # 生成缓存键
        img_array = np.array(image)

        # 新增：预处理图像
        processed_img = self.preprocess_image_for_segmentation(img_array, object_type)

        img_hash = hashlib.md5(
            cv2.resize(processed_img, (50, 50)).tobytes()
        ).hexdigest()

        cache_key = f"{img_hash}_{content_box}_{object_type.value}"

        # 检查缓存
        if cache_key in self._segmentation_cache:
            print(f"  ⚡ 使用缓存的分割结果")
            cached = self._segmentation_cache[cache_key]
            return cached['mask'], cached['score'], cached['method']

        # 设置图像（使用处理后的图像）
        self.predictor.set_image(processed_img)

        results = []

        # 策略1: 框提示
        if self.config.enable_multi_prompt:
            try:
                x, y, w, h = content_box
                margin = 5
                box_array = np.array([x + margin, y + margin,
                                      x + w - margin, y + h - margin])

                with torch.no_grad():
                    masks, scores, _ = self.predictor.predict(
                        box=box_array,
                        multimask_output=True
                    )

                best_idx = np.argmax(scores)
                results.append(('box', masks[best_idx], scores[best_idx]))
                print(f"  📦 框提示分数: {scores[best_idx]:.3f}")
            except Exception as e:
                print(f"  ❌ 框提示失败: {e}")

        # 策略2: 几何点提示（如果有点）
        if len(geometric_points) > 0:
            try:
                # 限制点数以加速
                max_points = 50
                if len(geometric_points) > max_points:
                    # 随机选择子集
                    indices = np.random.choice(len(geometric_points),
                                               max_points, replace=False)
                    selected_points = geometric_points[indices]
                else:
                    selected_points = geometric_points

                points = np.array([[p[1], p[0]] for p in selected_points])
                labels = np.ones(len(points))

                # 简化负样本点生成
                negative_points = self._generate_negative_points_fast(
                    processed_img.shape[:2], content_box, initial_mask
                )

                if len(negative_points) > 0:
                    neg_points = np.array([[p[1], p[0]] for p in negative_points])
                    points = np.vstack([points, neg_points])
                    labels = np.hstack([labels, np.zeros(len(neg_points))])

                with torch.no_grad():
                    masks, scores, _ = self.predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=True
                    )

                best_idx = np.argmax(scores)
                results.append(('geometric_points', masks[best_idx], scores[best_idx]))
                print(f"  🎯 几何点提示分数: {scores[best_idx]:.3f}")
            except Exception as e:
                print(f"  ❌ 几何点提示失败: {e}")

        # 策略3: 掩码引导（仅当初始掩码质量高时）
        if initial_mask is not None and self.config.enable_hierarchical:
            initial_quality = self._quick_quality_check(initial_mask)
            if initial_quality > 0.65:
                try:
                    h, w = processed_img.shape[:2]
                    kernel = np.ones((3, 3), np.uint8)
                    initial_mask_eroded = cv2.erode(
                        initial_mask.astype(np.uint8), kernel, iterations=1
                    )

                    mask_input = cv2.resize(
                        initial_mask_eroded.astype(np.float32), (w, h)
                    )
                    mask_input_resized = cv2.resize(
                        mask_input, (256, 256), interpolation=cv2.INTER_NEAREST
                    )

                    mask_input_tensor = torch.as_tensor(
                        mask_input_resized, device=self.sam_core.device
                    ).unsqueeze(0)

                    with torch.no_grad():
                        masks, scores, _ = self.predictor.predict(
                            mask_input=mask_input_tensor,
                            multimask_output=True
                        )

                    best_idx = np.argmax(scores)
                    results.append(('mask_guided', masks[best_idx], scores[best_idx]))
                    print(f"  🎭 掩码引导分数: {scores[best_idx]:.3f}")
                except Exception as e:
                    print(f"  ❌ 掩码引导失败: {e}")

        # 选择最佳结果
        if results:
            best_result = None
            best_composite_score = -1

            for method, mask, score in results:
                # 使用快速质量评估
                geometric_quality = self._quick_quality_check(mask)
                composite_score = score * 0.6 + geometric_quality * 0.4

                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_result = (method, mask, score)

            if best_result:
                method, mask, score = best_result
                print(f"  ✅ 选择最佳策略: {method}")

                # 缓存结果
                self._segmentation_cache[cache_key] = {
                    'mask': torch.from_numpy(mask),
                    'score': score,
                    'method': method
                }

                # 限制缓存大小
                if len(self._segmentation_cache) > 50:
                    keys_to_remove = list(self._segmentation_cache.keys())[:25]
                    for key in keys_to_remove:
                        del self._segmentation_cache[key]

                # 记录性能
                elapsed = time.time() - start_time
                self.timing_stats['segmentation'].append(elapsed)

                return torch.from_numpy(mask), score, method

        # 降级处理
        print("  🔄 降级使用初始掩码")
        full_mask = np.zeros(processed_img.shape[:2], dtype=bool)
        x, y, w, h = content_box
        if initial_mask is not None and initial_mask.shape == (h, w):
            full_mask[y:y + h, x:x + w] = initial_mask

        return torch.from_numpy(full_mask), 0.5, 'fallback'

    def _generate_negative_points_fast(self, shape: Tuple[int, int],
                                       content_box: Tuple[int, int, int, int],
                                       mask: np.ndarray) -> List[Tuple[int, int]]:
        """快速生成负样本点"""
        h, w = shape
        x, y, bw, bh = content_box

        # 只生成少量关键负样本点
        negative_points = []

        # 四个角点
        margin = 10
        corner_points = [
            (y - margin, x - margin),
            (y - margin, x + bw + margin),
            (y + bh + margin, x - margin),
            (y + bh + margin, x + bw + margin)
        ]

        for py, px in corner_points:
            if 0 <= py < h and 0 <= px < w:
                negative_points.append((py, px))

        # 四个边界中点
        edge_points = [
            (y - margin, x + bw // 2),
            (y + bh + margin, x + bw // 2),
            (y + bh // 2, x - margin),
            (y + bh // 2, x + bw + margin)
        ]

        for py, px in edge_points:
            if 0 <= py < h and 0 <= px < w:
                negative_points.append((py, px))

        return negative_points[:10]  # 最多10个负样本点

    # ============= 质量评估（优化版） =============

    def evaluate_mask_quality(self, mask: np.ndarray,
                              object_type: ObjectType) -> float:
        """评估掩码质量 - 带缓存"""
        start_time = time.time()

        if not np.any(mask):
            return 0.0

        # 检查缓存
        mask_hash = self._get_mask_hash(mask)
        cache_key = f"{mask_hash}_{object_type.value}"

        if cache_key in self._quality_cache:
            return self._quality_cache[cache_key]

        # 执行质量评估
        quality = self._compute_quality(mask, object_type)

        # 缓存结果
        self._quality_cache[cache_key] = quality

        # 限制缓存大小
        if len(self._quality_cache) > 100:
            keys_to_remove = list(self._quality_cache.keys())[:50]
            for key in keys_to_remove:
                del self._quality_cache[key]

        # 记录性能
        elapsed = time.time() - start_time
        self.timing_stats['quality_eval'].append(elapsed)

        return quality

    def _quick_quality_check(self, mask: np.ndarray) -> float:
        """快速质量检查 - 用于中间步骤"""
        if not np.any(mask):
            return 0.0

        mask_uint8 = mask.astype(np.uint8)

        # 只检查连通性
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

        if num_labels > 1:
            max_area = stats[1:, -1].max()
            total_area = mask.sum()
            if total_area > 0:
                return max_area / total_area

        return 0.8  # 单一连通组件

    def _compute_quality(self, mask: np.ndarray,
                         object_type: ObjectType) -> float:
        """计算完整质量分数"""
        mask_uint8 = mask.astype(np.uint8)

        # 连通性
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
        connectivity_score = 0
        if num_labels > 1:
            max_size = stats[1:, -1].max()
            connectivity_score = max_size / mask.sum() if mask.sum() > 0 else 0
        else:
            connectivity_score = 1.0

        # 紧凑性（简化计算）
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        compactness_score = 0.5  # 默认值

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            # 使用边界框近似代替凸包计算
            x, y, w, h = cv2.boundingRect(main_contour)
            bbox_area = w * h
            contour_area = cv2.contourArea(main_contour)
            if bbox_area > 0:
                compactness_score = contour_area / bbox_area

        base_score = connectivity_score * 0.6 + compactness_score * 0.4

        if self.config.use_geometric_quality and object_type in [ObjectType.CARVING, ObjectType.RELIEF]:
            # 简化的对称性检查
            symmetry_score = self._quick_symmetry_score(mask_uint8)
            final_score = base_score * 0.7 + symmetry_score * 0.3
        else:
            final_score = base_score

        return min(final_score, 1.0)

    def _quick_symmetry_score(self, mask: np.ndarray) -> float:
        """快速对称性评分"""
        h, w = mask.shape

        # 只检查垂直对称
        if w > 1:
            left_half = mask[:, :w // 2]
            right_half = np.fliplr(mask[:, w - (w // 2):])
            min_w = min(left_half.shape[1], right_half.shape[1])

            if min_w > 0:
                # 降采样以加速比较
                if h > 50:
                    left_small = cv2.resize(left_half[:, :min_w], (20, 20))
                    right_small = cv2.resize(right_half[:, :min_w], (20, 20))
                    return np.mean(left_small == right_small)
                else:
                    return np.mean(left_half[:, :min_w] == right_half[:, :min_w])

        return 0.5

    def print_performance_stats(self):
        """打印性能统计信息"""
        print("\n📊 性能统计:")
        for name, times in self.timing_stats.items():
            if times:
                avg_time = np.mean(times)
                max_time = np.max(times)
                min_time = np.min(times)
                print(f"  {name}:")
                print(f"    平均: {avg_time:.3f}秒")
                print(f"    最快: {min_time:.3f}秒")
                print(f"    最慢: {max_time:.3f}秒")

        print(f"\n💾 缓存使用:")
        print(f"  几何点缓存: {len(self._point_cache)}项")
        print(f"  分割缓存: {len(self._segmentation_cache)}项")
        print(f"  质量缓存: {len(self._quality_cache)}项")