"""
Beauty Filter Demo Script
对单张图像应用美颜效果的演示脚本
"""

import argparse
import sys
import os
import cv2
import numpy as np

# 导入依赖模块
from beauty_filter import BeautyFilter
from align import detect_face
import facenet

# 添加src路径到模块搜索路径
sys.path.insert(0, os.path.dirname(__file__))


def detect_faces_in_image(image, pnet, rnet, onet, minsize=20):
    """
    检测图像中的人脸和关键点

    Args:
        image: 输入图像
        pnet, rnet, onet: MTCNN网络
        minsize: 最小人脸大小

    Returns:
        人脸边界框列表 [(x, y, w, h), ...]
    """
    bounding_boxes, points = detect_face.detect_face(
        image, minsize, pnet, rnet, onet,
        threshold=[0.6, 0.7, 0.7], factor=0.709
    )

    rois = []
    for bbox in bounding_boxes:
        # bbox格式: [x1, y1, x2, y2, confidence]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        rois.append((x1, y1, x2, y2))

    return rois


def apply_beauty_to_image(image_path, output_path,
                          smoothness=1.0, brightness=0,
                          saturation=1.0, whitening=0,
                          use_face_detection=False):
    """
    对图像应用美颜效果

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        smoothness: 磨皮强度
        brightness: 亮度调整
        saturation: 饱和度
        whitening: 美白强度
        use_face_detection: 是否使用人脸检测（仅处理人脸区域）
    """
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return False

    print(f"Image size: {image.shape}")

    # 初始化美颜过滤器
    beauty_filter = BeautyFilter(
        smoothness=smoothness,
        brightness=brightness,
        saturation=saturation,
        whitening=whitening
    )

    print(f"\nApplying beauty filter...")
    print(f"  - Smoothness: {smoothness}")
    print(f"  - Brightness: {brightness}")
    print(f"  - Saturation: {saturation}")
    print(f"  - Whitening: {whitening}")

    result = image.copy()

    if use_face_detection:
        print("\nDetecting faces...")
        try:
            # 初始化MTCNN网络
            pnet, rnet, onet = detect_face.create_mtcnn(
                None, os.path.join(os.path.dirname(__file__), 'align')
            )

            # 检测人脸
            rois = detect_faces_in_image(image, pnet, rnet, onet)
            print(f"Found {len(rois)} face(s)")

            # 对每个人脸应用美颜
            for i, roi in enumerate(rois):
                print(f"  Processing face {i+1}...")
                result = beauty_filter.apply_beauty_filter(result, roi=roi)

        except Exception as e:
            print(f"Warning: Face detection failed ({e})")
            print("Applying beauty filter to entire image...")
            result = beauty_filter.apply_beauty_filter(image)
    else:
        # 对整个图像应用美颜
        result = beauty_filter.apply_beauty_filter(image)

    # 保存结果
    print(f"\nSaving result to: {output_path}")
    cv2.imwrite(output_path, result)

    # 显示对比
    print("Beauty filter applied successfully!")
    display_comparison(image, result)

    return True


def display_comparison(original, beauty):
    """
    显示原图和美颜后的对比

    Args:
        original: 原始图像
        beauty: 美颜后的图像
    """
    # 创建对比图像（左右并排）
    h, w = original.shape[:2]

    # 如果图像太大，缩小显示
    if w > 1280:
        scale = 1280 / w
        original = cv2.resize(original, (int(w * scale), int(h * scale)))
        beauty = cv2.resize(beauty, (int(w * scale), int(h * scale)))

    # 并排显示
    comparison = np.hstack([original, beauty])

    # 添加标签
    cv2.putText(comparison, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Beauty Filter", (original.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Comparison: Original vs Beauty Filter", comparison)
    print("\nPress any key to close the preview...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_arguments(argv):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Apply beauty filter to face images'
    )

    parser.add_argument('image_path',
                        help='Path to input image')
    parser.add_argument('--output', '-o',
                        help='Path to output image (default: input_beauty.jpg)')
    parser.add_argument('--smoothness', type=float, default=1.0,
                        help='Skin smoothing intensity (0.5-2.0). Default: 1.0')
    parser.add_argument('--brightness', type=int, default=0,
                        help='Brightness adjustment (-50-50). Default: 0')
    parser.add_argument('--saturation', type=float, default=1.0,
                        help='Skin color saturation (0.5-2.0). Default: 1.0')
    parser.add_argument('--whitening', type=int, default=0,
                        help='Whitening/brightening strength (0-50). Default: 0')
    parser.add_argument('--use-face-detection', action='store_true',
                        help='Use face detection (only beautify face regions)')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.image_path)
        output_path = f"{base}_beauty{ext}"

    # 应用美颜
    success = apply_beauty_to_image(
        args.image_path,
        output_path,
        smoothness=args.smoothness,
        brightness=args.brightness,
        saturation=args.saturation,
        whitening=args.whitening,
        use_face_detection=args.use_face_detection
    )

    sys.exit(0 if success else 1)
