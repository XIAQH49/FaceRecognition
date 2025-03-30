# AgeGender.py
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class AgeGenderDetector:
    def __init__(self, device="cpu"):
        # 加载模型文件（确保路径正确）
        self.face_net = cv2.dnn.readNet("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")
        self.age_net = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
        self.gender_net = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")
        self.device = device
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        self._set_backend()


    def _set_backend(self):
        # 配置计算后端
        if self.device == "gpu":
            backend = cv2.dnn.DNN_BACKEND_CUDA
            target = cv2.dnn.DNN_TARGET_CUDA
        else:
            backend = cv2.dnn.DNN_TARGET_CPU
            target = cv2.dnn.DNN_TARGET_CPU
        
        # 统一设置模型后端
        for net in [self.face_net, self.age_net, self.gender_net]:
            net.setPreferableBackend(backend)
            net.setPreferableTarget(target)

    def detect(self, frame):
        processed_frame, bboxes = self._get_face_boxes(frame)
        results = []  # 存储多个人脸的结果
        
        if bboxes:
            # 遍历所有检测到的人脸
            for idx, bbox in enumerate(bboxes):
                face = self._extract_face_roi(frame, bbox)
                gender = self._predict_gender(face)
                age = self._predict_age(face)
                # 生成标签（例如 "人脸1"）
                tag = f"人脸{idx+1}"
                results.append((tag, gender, age))
        
        return processed_frame, results  # 返回处理后的图像和结果列表

    def _get_face_boxes(self, frame, conf_threshold=0.7):
        # 人脸检测逻辑
        frame_height, frame_width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        bboxes = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                bboxes.append([x1, y1, x2, y2])
                # 绘制检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 添加标签
                tag = f"人脸{len(bboxes)}"
                # 计算字体高度
                font = ImageFont.truetype("simkai.ttf", 30)
                _, _, _, h = font.getbbox(tag)
                # 调整文本位置，使字体下对齐框
                pos = (x1, y1-30)
                # 使用 PIL 绘制中文标签
                frame = self._draw_text(frame, tag,  pos, font)
        
        return frame, bboxes
    
    def _draw_text(self, img, text, pos ,font):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=(0, 255, 0))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _extract_face_roi(self, frame, bbox, padding=20):
        # 提取人脸区域
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        return frame[y1:y2, x1:x2]

    def _predict_gender(self, face):
        # 性别预测
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()
        return self.gender_list[np.argmax(preds)]

    def _predict_age(self, face):
        # 年龄预测
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.age_net.setInput(blob)
        preds = self.age_net.forward()
        return self.age_list[np.argmax(preds)]