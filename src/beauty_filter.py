"""
Face Beauty Filter Module
使用OpenCV实现基础的人脸美颜效果（磨皮、肤色增强、亮度调整等）
"""

import cv2
import numpy as np


class BeautyFilter:
    """人脸美颜处理类"""

    def __init__(self,
                 smoothness=1.0,      # 磨皮强度 (0.5-2.0)
                 brightness=0,        # 亮度调整 (-50-50)
                 saturation=1.0,      # 饱和度 (0.5-2.0)
                 whitening=0):        # 提亮程度 (0-50)
        """
        初始化美颜参数
        Args:
            smoothness: 磨皮强度，越大越光滑 (默认1.0)
            brightness: 亮度调整，正数更亮 (默认0)
            saturation: 肤色饱和度 (默认1.0)
            whitening: 肤色提亮程度 (默认0)
        """
        self.smoothness = smoothness
        self.brightness = brightness
        self.saturation = saturation
        self.whitening = whitening

    def smooth_skin(self, image, kernel_size=15):
        """
        磨皮效果：使用双边滤波+高斯模糊混合

        Args:
            image: 输入图像 (BGR格式)
            kernel_size: 滤波核大小 (必须是奇数)

        Returns:
            磨皮后的图像
        """
        # 确保kernel_size是奇数
        kernel_size = int(kernel_size) if kernel_size % 2 == 1 else int(kernel_size) + 1

        # 双边滤波 - 保留边界的同时平滑颜色
        # 双边滤波的参数调整根据smoothness
        diameter = int(kernel_size * self.smoothness)
        diameter = diameter if diameter % 2 == 1 else diameter + 1
        diameter = max(5, min(diameter, 25))  # 限制范围在5-25之间

        bilateral = cv2.bilateralFilter(image, diameter, 80, 80)

        # 高斯模糊 - 进一步平滑
        gaussian = cv2.GaussianBlur(bilateral, (kernel_size, kernel_size), 0)

        # 混合：结合双边滤波和高斯模糊的效果
        # 权重比例根据smoothness调整
        alpha = min(self.smoothness / 2.0, 0.7)  # 确保不会过度平滑
        smoothed = cv2.addWeighted(bilateral, 1 - alpha, gaussian, alpha, 0)

        return smoothed

    def enhance_skin_tone(self, image):
        """
        肤色增强：提高肤色的饱和度和均匀性

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            肤色增强后的图像
        """
        # 转换到HSV颜色空间便于调整饱和度
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 调整饱和度
        hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        # 调整亮度 (V通道)
        hsv[:, :, 2] = hsv[:, :, 2] + self.whitening
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        # 转换回BGR
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return enhanced

    def adjust_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """
        亮度和对比度调整

        Args:
            image: 输入图像
            brightness: 亮度值 (-50-50)
            contrast: 对比度 (0.5-2.0)

        Returns:
            调整后的图像
        """
        if brightness == 0 and contrast == 1.0:
            return image

        # 调整亮度和对比度
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        return adjusted

    def apply_beauty_filter(self, image, roi=None):
        """
        应用完整的美颜效果

        Args:
            image: 输入图像 (BGR格式)
            roi: 感兴趣区域 (可选) - 只对ROI区域应用美颜
                格式: (x, y, w, h) 或 (x1, y1, x2, y2) 的边界框坐标

        Returns:
            美颜后的图像
        """
        # 如果指定了ROI，只对该区域应用美颜
        if roi is not None:
            # 复制原始图像
            result = image.copy()

            # 提取ROI (假设roi是 (x, y, w, h) 格式)
            if len(roi) == 4:
                x, y, w, h = roi
                roi_image = result[y:y+h, x:x+w]
            else:
                # 如果是边界框格式 (x1, y1, x2, y2)
                x1, y1, x2, y2 = roi
                roi_image = result[y1:y2, x1:x2]
                x, y, w, h = x1, y1, x2-x1, y2-y1

            # 对ROI应用美颜
            # 1. 磨皮
            smoothed = self.smooth_skin(roi_image)

            # 2. 肤色增强
            enhanced = self.enhance_skin_tone(smoothed)

            # 3. 亮度调整
            final = self.adjust_brightness_contrast(enhanced, self.brightness, 1.0)

            # 将处理结果放回原图像
            result[y:y+h, x:x+w] = final

            return result

        else:
            # 对整个图像应用美颜
            # 1. 磨皮
            smoothed = self.smooth_skin(image)

            # 2. 肤色增强
            enhanced = self.enhance_skin_tone(smoothed)

            # 3. 亮度调整
            final = self.adjust_brightness_contrast(enhanced, self.brightness, 1.0)

            return final


def apply_beauty_to_faces(image, faces, beauty_filter):
    """
    为图像中的每个人脸应用美颜效果

    Args:
        image: 输入图像
        faces: 人脸列表，每个人脸为 (x, y, w, h) 或 (x1, y1, x2, y2) 格式
        beauty_filter: BeautyFilter对象

    Returns:
        应用美颜后的图像
    """
    result = image.copy()

    for face in faces:
        result = beauty_filter.apply_beauty_filter(result, roi=face)

    return result
