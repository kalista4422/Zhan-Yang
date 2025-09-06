# geom_prompt_extractor/config.py

from dataclasses import dataclass
from enum import Enum

class ObjectType(Enum):
    """对象类型枚举"""
    CARVING = "carving"
    RELIEF = "relief"
    FLOWER = "flower"
    PRODUCT = "product"
    GENERAL = "general"

@dataclass
class ExtractionConfig:
    """提取配置参数"""
    # 内容框检测
    min_box_ratio: float = 0.25
    max_box_ratio: float = 0.98

    # SAM参数
    points_per_side: int = 48
    pred_iou_thresh: float = 0.85
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 30

    # 几何感知参数
    geometric_point_ratio: float = 0.6
    corner_detection_quality: float = 0.01
    curvature_threshold: float = 0.1

    # 多策略参数
    enable_multi_prompt: bool = True
    enable_hierarchical: bool = True

    # 质量评估参数
    min_quality_threshold: float = 0.7
    use_geometric_quality: bool = True

    # 后处理参数
    smooth_edges: bool = True
    adaptive_morphology: bool = True
    feather_radius: int = 5