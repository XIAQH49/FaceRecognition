# main_window.py
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from AgeGender import AgeGenderDetector

class VideoThread(QThread):
    update_frame = pyqtSignal(QImage, str, str)  # 发送帧图像、性别、年龄

    def __init__(self, device="cpu"):
        super().__init__()
        self.detector = AgeGenderDetector(device)
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(0)  # 默认摄像头
        while self.is_running:
            ret, frame = cap.read()
            if ret:
                # 处理帧并预测
                processed_frame, gender, age = self.detector.detect(frame)
                # 转换图像格式为 RGB
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # 发送信号
                self.update_frame.emit(qt_image, gender, age)
        cap.release()

    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_thread = VideoThread(device="cpu")  # 使用 CPU
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("人脸年龄性别识别系统")
        self.setGeometry(100, 100, 800, 600)
        
        # 视频显示区域
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # 结果显示区域
        self.result_label = QLabel("性别: Unknown\n年龄: Unknown", self)
        self.result_label.setStyleSheet("font-size: 20px; color: blue;")
        
        # 控制按钮
        self.btn_start = QPushButton("开始识别", self)
        self.btn_stop = QPushButton("停止识别", self)
        self.btn_start.clicked.connect(self.start_capture)
        self.btn_stop.clicked.connect(self.stop_capture)
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # 连接信号
        self.video_thread.update_frame.connect(self.update_display)

    def start_capture(self):
        self.video_thread.start()

    def stop_capture(self):
        self.video_thread.stop()

    def update_display(self, image, gender, age):
        # 更新视频帧和结果
        self.video_label.setPixmap(QPixmap.fromImage(image))
        self.result_label.setText(f"性别: {gender}\n年龄: {age}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())