# geom_prompt_extractor/extractor.py
"""
兼容性文件 - 导出主类
"""
from .post_processor import GeomPromptExtractorV8

# 保持向后兼容
__all__ = ['GeomPromptExtractorV8']