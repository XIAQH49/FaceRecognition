# interface_interaction.py
from main_window import MainWindow
from AgeGender import AgeGenderDetector
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import sys
from PyQt5 import QtWidgets
import os
import numpy as np  # 修复：添加 numpy 导入

class InterfaceInteraction:
    def __init__(self):
        self.main_window = MainWindow()
        self.detector = AgeGenderDetector()
        
        # 连接信号
        self.main_window.OpenCameraButton.clicked.connect(self.start_camera_and_detect)
        self.main_window.CloseCameraButton.clicked.connect(self.stop_camera)
        self.main_window.SelectButton.clicked.connect(self.select_file_and_detect)
        self.main_window.RecognizeButton.clicked.connect(self.real_time_recognize)

        # 初始化定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)

    def select_file_and_detect(self):
        #self.main_window.select_file()
        if self.main_window.selected_file_path:
            try:
                image = cv2.imread(self.main_window.selected_file_path)
                # 调用检测方法，获取多人脸结果
                processed_frame, face_results = self.detector.detect(image)
            
                # 显示处理后的图像（可选，如果需要标注框）
                self.main_window.display_image(processed_frame)
            
                # 清空旧数据
                self.main_window.model.removeRows(0, self.main_window.model.rowCount())
            
                # 添加多行数据到表格
                for tag, gender, age in face_results:
                    self.main_window.add_data_to_table(
                        self.main_window.selected_file_path,
                        tag,  # 标签（如 "人脸1"）
                        gender,
                        age
                    )
            except Exception as e:
                self.main_window._show_error_message(f"识别失败: {str(e)}", "错误")

    def start_camera_and_detect(self):
        if not self.main_window.cap:
            self.main_window.cap = cv2.VideoCapture(0)
            if self.main_window.cap.isOpened():
                self.timer.start(30)
                self.main_window.RecognizeButton.show()
            else:
                self.main_window._show_error_message("无法打开摄像头", "硬件错误")

    def update_camera_frame(self):
        if self.main_window.cap and self.main_window.cap.isOpened():
            ret, frame = self.main_window.cap.read()
            if ret:
                # 调用检测方法，获取处理后的帧和结果
                processed_frame, face_results = self.detector.detect(frame)

                # 将处理后的帧转换为RGB格式
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_image.data, w, h, bytes_per_line, 
                    QImage.Format_RGB888
                )
                self.main_window.image_display_label.setPixmap(
                    QPixmap.fromImage(qt_image).scaled(
                        self.main_window.image_display_label.size(),
                        Qt.KeepAspectRatioByExpanding,
                        Qt.SmoothTransformation
                    )
                )

    def real_time_recognize(self):
        if self.main_window.cap and self.timer.isActive():
            pixmap = self.main_window.image_display_label.pixmap()
            if pixmap:
                qimage = pixmap.toImage()
                width, height = qimage.width(), qimage.height()
                ptr = qimage.bits()
                ptr.setsize(qimage.byteCount())
                frame = np.array(ptr).reshape(height, width, 4)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
                # 调用检测方法，获取多人脸结果
                processed_frame, face_results = self.detector.detect(frame)
            
                # 清空旧数据
                self.main_window.model.removeRows(0, self.main_window.model.rowCount())
            
                # 添加多行数据到表格
                for tag, gender, age in face_results:
                    self.main_window.add_data_to_table(
                        "摄像头",  # 位置
                        tag,       # 标签（如 "人脸1"）
                        gender,
                        age
                    )
                # 显示处理后的图像
                self.main_window.display_image(processed_frame)

    def stop_camera(self):
        if self.main_window.cap:
            self.timer.stop()
            self.main_window.cap.release()
            self.main_window.cap = None
            self.main_window.image_display_label.clear()
            self.main_window.RecognizeButton.hide()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    interaction = InterfaceInteraction()
    interaction.main_window.show()
    sys.exit(app.exec_())