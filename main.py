
# 主程序入口：用于执行批量图像前景提取 - V8修复版

import os
import sys
from geom_prompt_extractor.extractor import GeomPromptExtractorV8  # 使用V8


def main():
    """主函数 - V8修复版"""

    # --- 1. 检查模型文件 ---
    checkpoint_path = "sam_vit_h_4b8939.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误：找不到模型文件 '{checkpoint_path}'")
        print("请从以下地址下载SAM ViT-H模型文件，并将其放置在此目录下:")
        print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        sys.exit(1)

    # --- 2. 创建并配置提取器实例 ---
    try:
        extractor = GeomPromptExtractorV8(  # 使用V8类
            checkpoint_path=checkpoint_path,
            model_type="vit_h",
            force_gpu=True
        )
    except Exception as e:
        print(f"❌ 初始化SAM模型失败: {e}")
        sys.exit(1)

    # --- 配置选项 ---
    DEBUG_MODE = False  # 设置为True以启用调试模式
    TEST_SINGLE = False  # 设置为True测试单个图像
    FAST_MODE = False  # 设置为True启用快速模式

    # 调试模式配置
    if DEBUG_MODE:
        print("\n⚠️ 调试模式已启用：")
        print("  - 降低采样密度")
        print("  - 禁用多策略分割")
        print("  - 限制几何框架数量")

        # 使用更保守的设置
        extractor.config.points_per_side = 32  # 减少采样密度
        extractor.config.enable_multi_prompt = False  # 禁用多提示
        extractor.config.enable_hierarchical = False  # 禁用层级分割
        extractor.config.min_mask_region_area = 50  # 增大最小区域

        # 限制几何框架检测
        extractor.sam_core.MAX_GEOMETRIC_FRAMES = 5  # 限制最大框架数
        extractor.sam_core.FRAME_DETECTION_TIMEOUT = 3  # 减少超时时间

    # 快速模式配置
    elif FAST_MODE:
        print("\n⚡ 快速模式已启用：")
        print("  - 降低质量换取速度")

        extractor.enable_fast_mode()

    # 正常模式配置（针对黑底雕刻品优化）
    else:
        print("\n🎯 使用黑底雕刻品优化配置:")

        # SAM核心参数 - 针对黑底图像调整
        extractor.config.points_per_side = 48  # 适中的采样密度
        extractor.config.pred_iou_thresh = 0.85  # 适中的IoU阈值
        extractor.config.stability_score_thresh = 0.92  # 稳定性要求
        extractor.config.min_mask_region_area = 20  # 适中的最小区域

        # 几何感知参数 - 增强边缘检测
        extractor.config.geometric_point_ratio = 0.7  # 70%几何点
        extractor.config.corner_detection_quality = 0.001  # 敏感的角点检测
        extractor.config.curvature_threshold = 0.05  # 适中的曲率阈值

        # 多策略参数
        extractor.config.enable_multi_prompt = True
        extractor.config.enable_hierarchical = True

        # 质量评估参数
        extractor.config.min_quality_threshold = 0.7  # 质量要求
        extractor.config.use_geometric_quality = True

        # 后处理参数
        extractor.config.smooth_edges = True
        extractor.config.adaptive_morphology = True
        extractor.config.feather_radius = 3  # 适中的羽化

        # 内容框检测参数
        extractor.config.min_box_ratio = 0.10  # 更小的最小框
        extractor.config.max_box_ratio = 0.99  # 最大框比例

        # 设置框架检测限制
        extractor.sam_core.MAX_GEOMETRIC_FRAMES = 10  # 限制框架数
        extractor.sam_core.FRAME_DETECTION_TIMEOUT = 5  # 5秒超时

        print(f"  - 几何点密度: {extractor.config.points_per_side}")
        print(f"  - 几何点比例: {extractor.config.geometric_point_ratio * 100:.0f}%")
        print(f"  - 稳定性阈值: {extractor.config.stability_score_thresh}")
        print(f"  - 质量阈值: {extractor.config.min_quality_threshold}")
        print(f"  - 最大几何框架: {extractor.sam_core.MAX_GEOMETRIC_FRAMES}")

    # --- 3. 定义输入和输出文件夹 ---
    input_folder = "input_images"
    output_folder = "batch_output_v8_fixed"  # 修复版输出文件夹

    # --- 4. 检查输入文件夹是否存在 ---
    if not os.path.exists(input_folder):
        print(f"❌ 错误：未找到输入文件夹 '{input_folder}'")
        print(f"请在当前目录下创建一个名为 '{input_folder}' 的文件夹，并将您要处理的图片放入其中。")

        # 尝试创建输入文件夹
        try:
            os.makedirs(input_folder)
            print(f"✅ 已自动创建输入文件夹 '{input_folder}'")
            print("请将您的图像文件放入该文件夹后重新运行程序。")
        except Exception as e:
            print(f"❌ 无法创建输入文件夹: {e}")

        sys.exit(1)

    # --- 5. 检查输入文件夹是否有图像 ---
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    input_files = [f for f in os.listdir(input_folder)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    if not input_files:
        print(f"⚠️ 警告：输入文件夹 '{input_folder}' 中没有找到图像文件")
        print(f"支持的图像格式: {', '.join(image_extensions)}")
        sys.exit(1)

    # --- 6. 测试单个图像（如果启用） ---
    if TEST_SINGLE:
        test_images = ["test (1).jpg", "test (1).png", "test (10).jpg"]  # 测试图像列表
        test_image_path = None

        for test_img in test_images:
            test_path = os.path.join(input_folder, test_img)
            if os.path.exists(test_path):
                test_image_path = test_path
                break

        if test_image_path:
            print(f"\n🔬 测试单个图像: {test_image_path}")
            print("=" * 70)

            result = extractor.process_with_extraction(
                test_image_path,
                output_folder,
                force_overwrite=True,
                timeout_seconds=30,  # 减少超时时间
                save_visualization=True
            )

            print("=" * 70)
            if result and result.get('success'):
                print(f"✅ 测试成功!")
                print(f"  - 对象类型: {result['object_type']}")
                print(f"  - IoU分数: {result['score']:.3f}")
                print(f"  - 质量评分: {result['quality']:.3f}")
                print(f"  - 处理时间: {result['process_time']:.2f}秒")
            else:
                print(f"❌ 测试失败: {result.get('error', '未知错误')}")

            return
        else:
            print(f"⚠️ 未找到测试图像")

    # --- 7. 执行批量处理 ---
    print(f"\n{'=' * 70}")
    print(f"🚀 开始V8批量处理（修复版）")
    print(f"📁 输入目录: '{input_folder}' ({len(input_files)}个文件)")
    print(f"📁 输出目录: '{output_folder}'")
    print(f"{'=' * 70}")

    # 批处理参数
    batch_params = {
        'input_dir': input_folder,
        'output_dir': output_folder,
        'force_overwrite': True,
        'batch_size': 3 if DEBUG_MODE else 5,  # 调试模式下减小批次
        'timeout_per_image': 30 if DEBUG_MODE else 60,  # 超时设置
        'save_visualizations': not FAST_MODE  # 快速模式下不保存可视化
    }

    # 执行批量提取
    results = extractor.batch_extract_optimized(**batch_params)

    # --- 8. 处理结果汇总 ---
    print(f"\n{'=' * 70}")
    print(f"🎉 V8批量处理完成（修复版）!")

    if results:
        # 统计成功和失败的数量
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]

        print(f"\n📊 处理统计:")
        print(f"  ✅ 成功: {len(successful)}/{len(results)}")
        print(f"  ❌ 失败: {len(failed)}/{len(results)}")

        if successful:
            # 计算平均分数
            import numpy as np
            avg_score = np.mean([r['score'] for r in successful])
            avg_quality = np.mean([r['quality'] for r in successful])
            avg_time = np.mean([r.get('process_time', 0) for r in successful if r.get('process_time')])

            print(f"\n📈 质量指标:")
            print(f"  - 平均IoU分数: {avg_score:.3f}")
            print(f"  - 平均质量评分: {avg_quality:.3f}")
            print(f"  - 平均处理时间: {avg_time:.2f}秒")

            # 按对象类型统计
            type_counts = {}
            for r in successful:
                obj_type = r.get('object_type', 'unknown')
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

            if type_counts:
                print(f"\n🏷️ 对象类型分布:")
                for obj_type, count in type_counts.items():
                    print(f"  - {obj_type}: {count}个")
                    # 计算每种类型的平均质量
                    type_results = [r for r in successful if r.get('object_type') == obj_type]
                    if type_results:
                        type_avg_quality = np.mean([r['quality'] for r in type_results])
                        print(f"    平均质量: {type_avg_quality:.3f}")

            # 找出最佳和最差的结果
            best_result = max(successful, key=lambda x: x['quality'])
            worst_result = min(successful, key=lambda x: x['quality'])

            print(f"\n🏆 最佳结果: {best_result['filename']} (质量: {best_result['quality']:.3f})")
            print(f"📉 最低质量: {worst_result['filename']} (质量: {worst_result['quality']:.3f})")

        if failed:
            print(f"\n⚠️ 失败的文件:")
            for f in failed[:10]:  # 最多显示10个
                error_msg = f.get('error', '未知错误')
                # 截断过长的错误信息
                if len(error_msg) > 100:
                    error_msg = error_msg[:97] + "..."
                print(f"  - {f['filename']}: {error_msg}")

            if len(failed) > 10:
                print(f"  ... 还有 {len(failed) - 10} 个失败文件")
    else:
        print("⚠️ 没有处理任何文件")

    print(f"\n💾 所有结果已保存到: '{output_folder}'")
    print(f"   - *_foreground.png: 提取的前景图像(透明背景)")
    if not FAST_MODE:
        print(f"   - *_v8_analysis.png: 处理过程可视化")

    # 打印性能统计（如果可用）
    if hasattr(extractor.seg_processor, 'print_performance_stats'):
        extractor.seg_processor.print_performance_stats()

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    # 确保Python环境可以找到geom_prompt_extractor包
    sys.path.append(os.getcwd())

    # 打印程序信息
    print("\n" + "=" * 70)
    print("    GeomPrompt Extractor V8 - 修复版前景提取工具")
    print("    特别优化：黑底雕刻品、浮雕、金银花纹等复杂对象")
    print("    修复内容：改进对象检测、限制框架数量、添加预处理")
    print("=" * 70)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断了程序运行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)