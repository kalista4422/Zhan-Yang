# geom_prompt_extractor/post_processor.py
"""
Part 3: 后处理与批量处理 - 性能优化版
包含：内存管理、超时机制、缓存优化、批量处理优化
"""

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import shutil
from typing import Tuple, List, Dict, Optional
import scipy.ndimage as ndimage
import time
import gc
import signal
from contextlib import contextmanager
import hashlib

# 从配置模块导入
from .config import ExtractionConfig, ObjectType
# 导入其他模块
from .sam_core import SAMCore
from .segmentation_processor import SegmentationProcessor

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 超时上下文管理器
@contextmanager
def timeout(seconds):
    """超时上下文管理器，防止处理卡死"""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"处理超过{seconds}秒超时")

    # Windows系统不支持SIGALRM，使用线程替代
    import platform
    if platform.system() == 'Windows':
        import threading
        timer = threading.Timer(seconds, lambda: None)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    else:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


class PostProcessor:
    """后处理器 - 处理掩码优化、前景提取和可视化"""

    def __init__(self, sam_core: SAMCore, config: ExtractionConfig = None):
        """初始化后处理器"""
        self.sam_core = sam_core
        self.config = config or ExtractionConfig()
        # 添加缓存字典
        self._processing_cache = {}

    def clear_cache(self):
        """清理处理缓存"""
        self._processing_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def post_process_mask_enhanced(self, mask: torch.Tensor,
                                   content_box: Tuple[int, int, int, int],
                                   object_type: ObjectType,
                                   method: str,
                                   image: np.ndarray = None,
                                   fast_mode: bool = False) -> torch.Tensor:
        """V8优化版后处理 - 添加快速模式选项"""
        mask_np = mask.cpu().numpy().astype(np.uint8)
        h, w = mask_np.shape
        x, y, bw, bh = content_box

        # 1. 严格的边界裁剪
        cleaned_mask = np.zeros_like(mask_np)
        margin = 1 if method == 'geometric_points' else 2
        x_start, y_start = max(0, x + margin), max(0, y + margin)
        x_end, y_end = min(w, x + bw - margin), min(h, y + bh - margin)

        if x_start < x_end and y_start < y_end:
            cleaned_mask[y_start:y_end, x_start:x_end] = mask_np[y_start:y_end, x_start:x_end]

        # 快速模式跳过复杂处理
        if fast_mode:
            return torch.from_numpy(cleaned_mask > 0)

        # 2. 对浮雕类型特殊处理
        if object_type == ObjectType.RELIEF:
            cleaned_mask = self._process_relief_mask(cleaned_mask, content_box)
        else:
            # 其他类型的边缘收缩
            if image is not None and len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)

                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(cleaned_mask, kernel, iterations=1)
                eroded = cv2.erode(cleaned_mask, kernel, iterations=1)
                mask_edge = dilated - eroded

                strong_edges = (edges > 100) & (mask_edge > 0)
                cleaned_mask[strong_edges] = 0

        # 3. 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                max_area_idx = np.argmax(areas) + 1

                cleaned_mask = np.zeros_like(cleaned_mask)
                cleaned_mask[labels == max_area_idx] = 1

                # 对于浮雕，保留更多的组件
                threshold = 0.03 if object_type == ObjectType.RELIEF else 0.05

                for i in range(1, num_labels):
                    if i == max_area_idx:
                        continue
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > bw * bh * threshold:
                        cx, cy = centroids[i]
                        center_dist = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
                        max_dist = 0.7 if object_type == ObjectType.RELIEF else 0.5
                        if center_dist < np.sqrt((w / 2) ** 2 + (h / 2) ** 2) * max_dist:
                            cleaned_mask[labels == i] = 1

        # 4. 形态学操作（可选）
        if self.config.adaptive_morphology and not fast_mode:
            if object_type == ObjectType.CARVING:
                kernel_tiny = np.ones((2, 2), np.uint8)
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_tiny, iterations=1)

                filled = ndimage.binary_fill_holes(cleaned_mask).astype(np.uint8)

                diff = filled - cleaned_mask
                if np.sum(diff) < bw * bh * 0.1:
                    cleaned_mask = filled

                kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)

            elif object_type == ObjectType.RELIEF:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                cleaned_mask = ndimage.binary_fill_holes(cleaned_mask).astype(np.uint8)
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

            else:
                kernel = np.ones((2, 2), np.uint8)
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 5. 最终的边缘平滑（可选）
        if self.config.smooth_edges and not fast_mode:
            cleaned_mask_float = cleaned_mask.astype(np.float32)
            if object_type == ObjectType.RELIEF:
                smoothed = cv2.bilateralFilter(cleaned_mask_float, 5, 75, 75)
                cleaned_mask = (smoothed > 0.3).astype(np.uint8)
            else:
                smoothed = cv2.bilateralFilter(cleaned_mask_float, 3, 50, 50)
                cleaned_mask = (smoothed > 0.5).astype(np.uint8)

        return torch.from_numpy(cleaned_mask > 0)

    def _process_relief_mask(self, mask: np.ndarray,
                             content_box: Tuple[int, int, int, int]) -> np.ndarray:
        """专门处理浮雕掩码"""
        x, y, bw, bh = content_box

        # 使用距离变换找到主体区域
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # 找到距离变换的峰值点作为种子点
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

        # 使用分水岭算法改善分割
        if max_val > 0:
            markers = np.zeros_like(mask)
            markers[max_loc[1], max_loc[0]] = 1
            markers[0, :] = 2
            markers[-1, :] = 2
            markers[:, 0] = 2
            markers[:, -1] = 2

            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(mask_3channel, markers.astype(np.int32))
            refined_mask = (markers == 1).astype(np.uint8)
            combined_mask = np.logical_or(mask > 0, refined_mask > 0).astype(np.uint8)

            return combined_mask

        return mask

    def extract_foreground(self, image: Image.Image, mask: torch.Tensor,
                           content_box: Tuple[int, int, int, int] = None,
                           crop_to_content: bool = True) -> Tuple[np.ndarray, int]:
        """提取前景"""
        img_array = np.array(image)
        mask_array = mask.cpu().numpy()

        # 检测对象类型用于调整羽化
        object_type = self.sam_core.detect_object_type(Image.fromarray(img_array))

        if crop_to_content and content_box is not None:
            x, y, bw, bh = content_box
            cropped_img = img_array[y:y + bh, x:x + bw]
            cropped_mask = mask_array[y:y + bh, x:x + bw]

            alpha = (cropped_mask * 255).astype(np.uint8)
            if self.config.smooth_edges:
                feather_radius = 3 if object_type == ObjectType.RELIEF else self.config.feather_radius
                alpha = cv2.GaussianBlur(alpha,
                                         (feather_radius * 2 + 1,
                                          feather_radius * 2 + 1), 0)

            foreground_rgba = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2RGBA)
            foreground_rgba[:, :, 3] = alpha
            return foreground_rgba, np.sum(cropped_mask)
        else:
            alpha = (mask_array * 255).astype(np.uint8)
            foreground_rgba = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
            foreground_rgba[:, :, 3] = alpha
            return foreground_rgba, np.sum(mask_array)

    def visualize_results(self, image: Image.Image,
                          content_box: Tuple[int, int, int, int],
                          initial_mask: np.ndarray,
                          geometric_points: np.ndarray,
                          final_mask: torch.Tensor,
                          score: float,
                          object_type: ObjectType,
                          method: str,
                          filename: str,
                          output_path: Path,
                          save_visualization: bool = True):
        """创建可视化结果 - 添加保存控制选项"""
        if not save_visualization:
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        img_array = np.array(image)
        x, y, w, h = content_box

        # 1. Content Box
        axes[0, 0].imshow(image)
        rect = patches.Rectangle((x, y), w, h, lw=3, ec='yellow', fc='none', ls='--')
        axes[0, 0].add_patch(rect)
        axes[0, 0].set_title(f'内容框检测 (类型: {object_type.value})')

        # 2. Initial Mask
        axes[0, 1].imshow(image)
        if initial_mask is not None and initial_mask.any():
            full_initial_mask = np.zeros_like(img_array, dtype=np.uint8)
            initial_mask_resized = cv2.resize(initial_mask.astype(np.uint8), (w, h))
            full_initial_mask[y:y + h, x:x + w, 0] = initial_mask_resized * 255
            axes[0, 1].imshow(full_initial_mask, alpha=0.5)
        axes[0, 1].set_title('智能碎片聚合')

        # 3. Geometric Points
        axes[0, 2].imshow(image)
        if geometric_points is not None and len(geometric_points) > 0:
            points_for_display = np.array([[p[1] + x, p[0] + y] for p in geometric_points])
            axes[0, 2].scatter(points_for_display[:, 0], points_for_display[:, 1],
                               color='lime', s=25, marker='*')
        axes[0, 2].set_title('V8几何感知点分布')

        # 4. Final Mask Overlay
        axes[1, 0].imshow(image)
        final_mask_overlay = np.zeros_like(img_array)
        final_mask_overlay[final_mask.cpu().numpy()] = [138, 43, 226]
        axes[1, 0].imshow(final_mask_overlay, alpha=0.6)
        axes[1, 0].set_title(f'多策略SAM分割 ({method}, IoU: {score:.3f})')

        # 5. Final Mask
        axes[1, 1].imshow(final_mask.cpu().numpy(), cmap='gray')
        title = 'V8自适应后处理掩码'
        if object_type == ObjectType.RELIEF:
            title += ' (浮雕优化)'
        axes[1, 1].set_title(title)

        # 6. Extracted Foreground
        foreground_only, _ = self.extract_foreground(image, final_mask, content_box, crop_to_content=True)
        axes[1, 2].imshow(foreground_only)
        axes[1, 2].set_title('前景提取结果')

        for ax in axes.ravel():
            ax.axis('off')

        plt.suptitle(f'V8 Final: {filename}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path / f"{filename}_v8_analysis.png", dpi=120)
        plt.close('all')  # 确保关闭所有图形

        # 清理matplotlib缓存
        plt.clf()
        plt.cla()
        gc.collect()


class GeomPromptExtractorV8:
    """GeomPrompt V8主类 - 性能优化版"""

    def __init__(self, checkpoint_path: str, model_type: str = 'vit_h',
                 device: str = None, force_gpu: bool = True):
        """初始化V8提取器"""
        # 初始化核心模块
        self.sam_core = SAMCore(checkpoint_path, model_type, device, force_gpu)
        self.config = self.sam_core.config

        # 初始化处理器
        self.seg_processor = SegmentationProcessor(self.sam_core, self.config)
        self.post_processor = PostProcessor(self.sam_core, self.config)

        # 兼容性属性
        self.sam = self.sam_core.sam
        self.predictor = self.sam_core.predictor
        self.device = self.sam_core.device

        # 性能统计
        self.processing_times = []

        # 快速模式标志
        self.fast_mode = False

    def enable_fast_mode(self):
        """启用快速模式"""
        self.fast_mode = True
        self.config.points_per_side = 32
        self.config.enable_multi_prompt = False
        self.config.smooth_edges = False
        self.config.adaptive_morphology = False
        print("⚡ 快速模式已启用")

    def disable_fast_mode(self):
        """禁用快速模式"""
        self.fast_mode = False
        self.config = ExtractionConfig()  # 恢复默认配置
        print("🔧 快速模式已禁用")

    def adjust_config_for_object_type(self, object_type: ObjectType):
        """根据对象类型自动调整配置"""
        if self.fast_mode:
            return  # 快速模式下不调整

        if object_type == ObjectType.RELIEF:
            print("  📋 应用浮雕优化配置...")
            self.config.points_per_side = 84
            self.config.pred_iou_thresh = 0.82
            self.config.stability_score_thresh = 0.88
            self.config.min_mask_region_area = 8
            self.config.geometric_point_ratio = 0.9
            self.config.corner_detection_quality = 0.0003
            self.config.curvature_threshold = 0.02
            self.config.min_quality_threshold = 0.65
            self.config.feather_radius = 3

    def adjust_points_by_size(self, image_size: Tuple[int, int]) -> int:
        """根据图像大小动态调整采样点数量"""
        w, h = image_size
        area = w * h

        if self.fast_mode:
            return 40  # 快速模式固定40个点

        if area < 500 * 500:
            return 80
        elif area < 1000 * 1000:
            return 120
        else:
            return 150

    def process_with_extraction(self, image_path: str,
                                output_dir: str = "extraction_results_v8",
                                force_overwrite: bool = True,
                                timeout_seconds: int = 60,
                                save_visualization: bool = True) -> Optional[Dict]:
        """处理单张图像 - 添加超时和内存管理"""
        start_time = time.time()
        filename = Path(image_path).stem

        # 检查是否已处理
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        foreground_file = output_path / f"{filename}_foreground.png"
        if not force_overwrite and foreground_file.exists():
            print(f"⚠️ 文件 {filename} 已存在，跳过。")
            return None

        print(f"\n📷 GeomPrompt前景提取 V8: {filename}")

        try:
            # 使用超时机制
            with timeout(timeout_seconds):
                # 加载图像
                image = Image.open(image_path).convert("RGB")
                print(f"   尺寸: {image.width}x{image.height}")

                # 检测对象类型
                object_type = self.sam_core.detect_object_type(image)
                print(f"🔍 检测到对象类型: {object_type.value}")

                # 根据类型调整配置
                self.adjust_config_for_object_type(object_type)

                # 根据图像大小调整点数
                n_points = self.adjust_points_by_size((image.width, image.height))

                # 检测内容框
                content_box = self.sam_core.detect_content_box(image, object_type)

                # 创建初始掩码
                initial_mask_roi = self.sam_core.create_initial_mask(image, content_box, object_type)

                # 生成几何点
                geometric_points = self.seg_processor.generate_geometric_points(
                    initial_mask_roi, object_type, n_points=n_points
                )

                # 多提示SAM分割
                final_mask, score, method = self.seg_processor.multi_prompt_sam_segmentation(
                    image, content_box, initial_mask_roi, geometric_points, object_type
                )

                # V8增强后处理
                img_array = np.array(image)
                final_mask = self.post_processor.post_process_mask_enhanced(
                    final_mask, content_box, object_type, method, img_array,
                    fast_mode=self.fast_mode
                )

                # 评估质量
                final_quality = self.seg_processor.evaluate_mask_quality(
                    final_mask.cpu().numpy(), object_type
                )
                print(f"📊 最终质量评分: {final_quality:.3f}")

                # 提取前景
                foreground_only, pixels = self.post_processor.extract_foreground(
                    image, final_mask, content_box
                )
                Image.fromarray(foreground_only, 'RGBA').save(foreground_file)

                # 可视化结果（如果需要）
                if save_visualization:
                    self.post_processor.visualize_results(
                        image, content_box, initial_mask_roi, geometric_points,
                        final_mask, score, object_type, method, filename, output_path,
                        save_visualization=save_visualization
                    )

                # 记录处理时间
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                print(f"✅ 完成! 耗时: {process_time:.2f}秒")

                # 清理内存
                del image, img_array, initial_mask_roi, geometric_points, final_mask
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                return {
                    'filename': filename,
                    'success': True,
                    'score': score,
                    'quality': final_quality,
                    'object_type': object_type.value,
                    'process_time': process_time
                }

        except TimeoutError as e:
            print(f"⏰ {filename} 处理超时 ({timeout_seconds}秒)")
            return {
                'filename': filename,
                'success': False,
                'error': str(e)
            }
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'filename': filename,
                'success': False,
                'error': str(e)
            }
        finally:
            # 确保清理内存
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def batch_extract_optimized(self, input_dir: str = "input_images",
                                output_dir: str = "batch_output_v8",
                                force_overwrite: bool = True,
                                batch_size: int = 5,
                                timeout_per_image: int = 60,
                                save_visualizations: bool = True) -> List[Dict]:
        """优化的批量处理 - 分批处理避免内存溢出"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if force_overwrite and output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取图像文件列表
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = sorted([p for p in input_path.iterdir()
                              if p.suffix.lower() in image_extensions])

        if not image_files:
            print(f"❌ 在 '{input_dir}' 中未找到图像文件")
            return []

        print(f"🔍 找到 {len(image_files)} 个图像文件")
        print(f"⚙️ 批处理配置: 批大小={batch_size}, 超时={timeout_per_image}秒/图")

        results = []
        total_files = len(image_files)

        # 分批处理
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = image_files[batch_start:batch_end]

            print(f"\n{'=' * 60}")
            print(f"📦 处理批次 {batch_start // batch_size + 1}/{(total_files - 1) // batch_size + 1}")
            print(f"   文件 {batch_start + 1}-{batch_end}/{total_files}")
            print(f"{'=' * 60}")

            # 处理当前批次
            for i, image_file in enumerate(batch_files, batch_start + 1):
                print(f"\n[{i}/{total_files}] " + "=" * 50)

                result = self.process_with_extraction(
                    str(image_file),
                    output_path,
                    force_overwrite,
                    timeout_seconds=timeout_per_image,
                    save_visualization=save_visualizations
                )

                if result:
                    results.append(result)

            # 每批次后清理内存
            print(f"\n🧹 清理批次内存...")
            self.post_processor.clear_cache()
            gc.collect()

            if self.device == "cuda":
                torch.cuda.empty_cache()
                # 显示GPU内存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024 ** 3
                    reserved = torch.cuda.memory_reserved() / 1024 ** 3
                    print(f"   GPU内存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")

        # 打印最终汇总
        self._print_summary(results, total_files)

        return results

    def _print_summary(self, results: List[Dict], total_files: int):
        """打印处理汇总信息"""
        successful = sum(1 for r in results if r.get('success'))
        print(f"\n{'=' * 70}")
        print(f"📊 批处理完成: 成功 {successful}/{total_files}")

        if successful > 0:
            successful_results = [r for r in results if r.get('success')]

            # 按类型统计
            type_counts = {}
            for r in successful_results:
                obj_type = r.get('object_type', 'unknown')
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

            print(f"\n📈 对象类型分布:")
            for obj_type, count in type_counts.items():
                print(f"   {obj_type}: {count}个")

            # 计算平均指标
            avg_score = np.mean([r['score'] for r in successful_results])
            avg_quality = np.mean([r['quality'] for r in successful_results])
            avg_time = np.mean([r.get('process_time', 0) for r in successful_results if r.get('process_time')])

            print(f"\n📊 性能指标:")
            print(f"   平均IoU分数: {avg_score:.3f}")
            print(f"   平均质量评分: {avg_quality:.3f}")
            print(f"   平均处理时间: {avg_time:.2f}秒")

            # 特别统计各类型
            for obj_type in type_counts.keys():
                type_results = [r for r in successful_results if r.get('object_type') == obj_type]
                if type_results:
                    type_avg_score = np.mean([r['score'] for r in type_results])
                    type_avg_quality = np.mean([r['quality'] for r in type_results])
                    print(f"\n   {obj_type}类型:")
                    print(f"      平均IoU: {type_avg_score:.3f}")
                    print(f"      平均质量: {type_avg_quality:.3f}")

        # 失败统计
        failed = [r for r in results if not r.get('success')]
        if failed:
            print(f"\n⚠️ 失败的文件 ({len(failed)}个):")
            for f in failed[:5]:  # 只显示前5个
                print(f"   - {f['filename']}: {f.get('error', '未知错误')}")
            if len(failed) > 5:
                print(f"   ... 还有 {len(failed) - 5} 个失败文件")

        print(f"{'=' * 70}")

    # ============= 兼容性方法 =============

    def batch_extract(self, input_dir: str = "input_images",
                      output_dir: str = "batch_output_v8",
                      force_overwrite: bool = True) -> List[Dict]:
        """兼容性方法 - 调用优化版批处理"""
        return self.batch_extract_optimized(
            input_dir=input_dir,
            output_dir=output_dir,
            force_overwrite=force_overwrite,
            batch_size=5,
            timeout_per_image=60,
            save_visualizations=True
        )

    def detect_content_box_enhanced(self, image, object_type):
        """兼容性方法"""
        return self.sam_core.detect_content_box(image, object_type)

    def create_adaptive_mask_for_carving(self, image, content_box, object_type):
        """兼容性方法"""
        return self.sam_core.create_initial_mask(image, content_box, object_type)

    def generate_geometric_points(self, mask, object_type, n_points=120):
        """兼容性方法"""
        return self.seg_processor.generate_geometric_points(mask, object_type, n_points)

    def multi_prompt_sam_segmentation(self, image, content_box, initial_mask,
                                      geometric_points, object_type):
        """兼容性方法"""
        return self.seg_processor.multi_prompt_sam_segmentation(
            image, content_box, initial_mask, geometric_points, object_type
        )

    def post_process_mask_enhanced(self, mask, content_box, object_type, method, image=None):
        """兼容性方法"""
        return self.post_processor.post_process_mask_enhanced(
            mask, content_box, object_type, method, image, fast_mode=self.fast_mode
        )

    def extract_foreground(self, image, mask, content_box=None, crop_to_content=True):
        """兼容性方法"""
        return self.post_processor.extract_foreground(image, mask, content_box, crop_to_content)

    def visualize_results(self, *args, **kwargs):
        """兼容性方法"""
        return self.post_processor.visualize_results(*args, **kwargs)