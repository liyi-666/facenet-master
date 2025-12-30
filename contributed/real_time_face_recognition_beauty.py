# coding=utf-8
"""
Performs face detection and beauty filter in realtime.

扩展自原始 real_time_face_recognition.py，添加人脸美颜效果
"""
import argparse
import sys
import time
import os

import cv2

# 导入face和美颜模块
import face
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from beauty_filter import BeautyFilter, apply_beauty_to_faces


def add_overlays(frame, faces, frame_rate, show_fps=True):
    """
    在frame上绘制人脸框和标签

    Args:
        frame: 输入图像
        faces: 人脸列表
        frame_rate: 帧率
        show_fps: 是否显示FPS
    """
    if faces is not None:
        for face_obj in faces:
            face_bb = face_obj.bounding_box.astype(int)
            # 绘制人脸边界框 (绿色)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            # 绘制人脸名称标签
            if face_obj.name is not None:
                cv2.putText(frame, face_obj.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    if show_fps:
        # 显示帧率
        cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)


def extract_face_rois(faces):
    """
    从人脸对象列表提取ROI坐标

    Args:
        faces: 人脸列表

    Returns:
        ROI列表 [(x1, y1, x2, y2), ...]
    """
    rois = []
    if faces is not None:
        for face_obj in faces:
            face_bb = face_obj.bounding_box.astype(int)
            rois.append((face_bb[0], face_bb[1], face_bb[2], face_bb[3]))
    return rois


def main(args):
    """
    主函数：实时人脸识别+美颜

    Args:
        args: 命令行参数
    """
    # 参数设置
    frame_interval = 3  # 检测间隔帧数
    fps_display_interval = 5  # FPS显示间隔（秒）
    frame_rate = 0
    frame_count = 0

    # 初始化视频捕获和人脸识别
    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()

    # 初始化美颜过滤器
    beauty_filter = BeautyFilter(
        smoothness=args.smoothness,      # 磨皮强度
        brightness=args.brightness,      # 亮度
        saturation=args.saturation,      # 饱和度
        whitening=args.whitening         # 美白强度
    )

    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    print("=== Real-time Face Recognition with Beauty Filter ===")
    print(f"Beauty Settings:")
    print(f"  - Smoothness (Skin Smoothing): {args.smoothness}")
    print(f"  - Brightness: {args.brightness}")
    print(f"  - Saturation: {args.saturation}")
    print(f"  - Whitening: {args.whitening}")
    print(f"\nPress 'q' to quit")
    print(f"Press 'b' to toggle beauty filter")
    print(f"Press 's'/'d' to increase/decrease smoothness")
    print(f"Press 'w'/'e' to increase/decrease whitening")
    print("=" * 50)

    beauty_enabled = not args.no_beauty  # 默认启用美颜

    while True:
        # 逐帧捕获
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to read frame")
            break

        # 每frame_interval帧进行一次人脸检测
        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # 计算和显示FPS
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        # 应用美颜效果（如果启用）
        if beauty_enabled and faces is not None:
            rois = extract_face_rois(faces)
            if rois:
                frame = apply_beauty_to_faces(frame, rois, beauty_filter)

        # 绘制检测结果（人脸框和标签）
        add_overlays(frame, faces, frame_rate, show_fps=True)

        # 显示美颜状态
        status_text = "Beauty: ON" if beauty_enabled else "Beauty: OFF"
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    thickness=2, lineType=2)

        frame_count += 1
        cv2.imshow('Real-time Face Recognition with Beauty Filter', frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('b'):
            # 切换美颜开关
            beauty_enabled = not beauty_enabled
            status = "enabled" if beauty_enabled else "disabled"
            print(f"Beauty filter {status}")
        elif key == ord('s'):
            # 增加磨皮强度
            beauty_filter.smoothness = min(beauty_filter.smoothness + 0.2, 2.0)
            print(f"Smoothness increased to {beauty_filter.smoothness:.1f}")
        elif key == ord('d'):
            # 减少磨皮强度
            beauty_filter.smoothness = max(beauty_filter.smoothness - 0.2, 0.5)
            print(f"Smoothness decreased to {beauty_filter.smoothness:.1f}")
        elif key == ord('w'):
            # 增加美白强度
            beauty_filter.whitening = min(beauty_filter.whitening + 5, 50)
            print(f"Whitening increased to {beauty_filter.whitening}")
        elif key == ord('e'):
            # 减少美白强度
            beauty_filter.whitening = max(beauty_filter.whitening - 5, 0)
            print(f"Whitening decreased to {beauty_filter.whitening}")

    # 清理资源
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Real-time Face Recognition with Beauty Filter'
    )

    # 原始参数
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug outputs.')

    # 美颜参数
    parser.add_argument('--smoothness', type=float, default=1.0,
                        help='Skin smoothing intensity (0.5-2.0). Default: 1.0')
    parser.add_argument('--brightness', type=int, default=0,
                        help='Brightness adjustment (-50-50). Default: 0')
    parser.add_argument('--saturation', type=float, default=1.0,
                        help='Skin color saturation (0.5-2.0). Default: 1.0')
    parser.add_argument('--whitening', type=int, default=0,
                        help='Whitening/brightening strength (0-50). Default: 0')
    parser.add_argument('--no-beauty', action='store_true',
                        help='Disable beauty filter by default.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
