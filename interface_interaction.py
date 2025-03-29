from main_window import MainWindow
from AgeGender import AgeGenderDetector
from PyQt5.QtGui import QStandardItem
import cv2
import sys
from PyQt5 import QtWidgets


class InterfaceInteraction:
    def __init__(self):
        # 创建主窗口实例
        self.main_window = MainWindow()
        # 创建年龄性别检测器实例
        self.detector = AgeGenderDetector()

        # 连接按钮点击信号到对应的处理方法
        self.main_window.SelectButton.clicked.connect(self.select_file_and_detect)
        self.main_window.StartButton.clicked.connect(self.start_camera_and_detect)
        self.main_window.StopButton.clicked.connect(self.stop_camera)

    def select_file_and_detect(self):
        # 调用主窗口的选择文件方法
        self.main_window.select_file()
        file_path = self.main_window.selected_file_path
        if file_path:
            try:
                # 读取图片
                image = cv2.imread(file_path)
                # 调用年龄性别检测器的检测方法
                processed_frame, gender, age = self.detector.detect(image)
                # 在主窗口显示处理后的图片
                self.main_window.display_image(file_path)
                # 更新表格数据
                self.update_table(file_path, gender, age)
            except Exception as e:
                print(f"处理图片时出错: {e}")

    def start_camera_and_detect(self):
        if self.main_window.cap is None:
            self.main_window.cap = cv2.VideoCapture(0)
            while True:
                ret, frame = self.main_window.cap.read()
                if ret:
                    # 调用年龄性别检测器的检测方法
                    processed_frame, gender, age = self.detector.detect(frame)
                    # 将OpenCV图像转换为Qt图像
                    height, width, channel = processed_frame.shape
                    bytesPerLine = 3 * width
                    qImg = self.main_window.QImage(processed_frame.data, width, height, bytesPerLine,
                                                   self.main_window.QImage.Format_RGB888).rgbSwapped()
                    pixmap = self.main_window.QPixmap.fromImage(qImg)
                    pixmap = pixmap.scaled(self.main_window.image_display_label.size(),
                                           aspectRatioMode=self.main_window.Qt.KeepAspectRatioByExpanding,
                                           transformMode=self.main_window.Qt.SmoothTransformation)
                    # 在主窗口的QLabel中显示图像
                    self.main_window.image_display_label.setPixmap(pixmap)
                    # 处理Qt事件
                    self.main_window.QtWidgets.QApplication.processEvents()
                    # 更新表格数据
                    self.update_table("摄像头", gender, age)
                if cv2.waitKey(1) & 0xFF == ord('q') or self.main_window.cap is None:
                    break
            cv2.destroyAllWindows()

    def stop_camera(self):
        # 调用主窗口的停止摄像头方法
        self.main_window.stop_camera()

    def update_table(self, location, gender, age):
        # 创建一行数据
        row = []
        for item in [location, gender, age]:
            cell = QStandardItem(item)
            row.append(cell)
        # 将数据行添加到表格模型中
        self.main_window.model.appendRow(row)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    interaction = InterfaceInteraction()
    interaction.main_window.show()
    sys.exit(app.exec_())