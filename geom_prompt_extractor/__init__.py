# geom_prompt_extractor/__init__.py
"""
GeomPrompt Extractor V8 - 几何感知前景提取工具
模块化版本，分为三个核心组件
"""

from .config import ExtractionConfig, ObjectType
from .post_processor import GeomPromptExtractorV8
from .sam_core import SAMCore
from .segmentation_processor import SegmentationProcessor
from .post_processor import PostProcessor

# 版本信息
__version__ = "8.0.0"
__author__ = "GeomPrompt Team"

# 导出的类和函数
__all__ = [
    'GeomPromptExtractorV8',
    'ExtractionConfig',
    'ObjectType',
    'SAMCore',
    'SegmentationProcessor', 
    'PostProcessor'
]

# 简化的导入别名
Extractor = GeomPromptExtractorV8