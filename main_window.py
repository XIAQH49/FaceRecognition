from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QImage, QPixmap
from PyQt5.QtCore import Qt
import os
import cv2

# 加载UI文件
Ui_MainWindow, _ = uic.loadUiType('main_window.ui')

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 初始化QTableView，假设objectName为OutputView
        self.table_view = self.OutputView
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["位置", "标签","性别", "年龄"])  # 设置表头
        self.table_view.setModel(self.model)

        # 设置列宽
        self._set_table_column_widths()

        # 连接按钮的clicked信号到相应的槽函数
        self._connect_button_signals()

        # 存储选择的文件路径
        self.selected_file_path = None
        # 初始化摄像头对象
        self.cap = None
        # 获取用于显示图像的QLabel
        self.image_display_label = self.DisplayLabel

        # 设置QLabel的边框为红色
        self._set_image_label_border()

        # 设置QLabel的对齐方式为居中
        self.image_display_label.setAlignment(Qt.AlignCenter)

        # 初始隐藏识别按钮
        self.RecognizeButton.hide()

    def _set_table_column_widths(self):
        """设置表格列宽"""
        self.table_view.setColumnWidth(0, 440)
        self.table_view.setColumnWidth(1, 58)
        self.table_view.setColumnWidth(2, 150)
        self.table_view.setColumnWidth(3, 150)

    def _connect_button_signals(self):
        """连接按钮的clicked信号到相应的槽函数"""
        self.SelectButton.clicked.connect(self.select_file)
        self.OpenCameraButton.clicked.connect(self.start_camera)
        self.CloseCameraButton.clicked.connect(self.stop_camera)
        self.RecognizeButton.clicked.connect(self.recognize)

    def _set_image_label_border(self):
        """设置图像显示标签的边框样式"""
        border_style = "border: 2px solid red;"
        self.image_display_label.setStyleSheet(border_style)

    def select_file(self):
        file_dialog = QtWidgets.QFileDialog()
        # 添加文件类型限制，这里以只允许选择图片文件为例
        file_filter = "图片文件 (*.png *.jpg *.jpeg);;所有文件 (*.*)"
        file_path, _ = file_dialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            allowed_extensions = ['.png', '.jpg', '.jpeg']
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension not in allowed_extensions:
                # 创建并显示QMessageBox提示框
                self._show_error_message(f"不允许选择 {file_extension} 类型的文件，请选择图片文件。", "文件类型错误")
            else:
                print(f"选择的文件路径: {file_path}")
                self.selected_file_path = file_path
                # 显示选择的图片
                self.display_image(file_path)
                # 模拟识别结果，添加数据到表格
                self.add_data_to_table(file_path,"未标记","男", "25")
                # 释放不必要的内存
                self._release_unused_memory()

    def _show_error_message(self, message, title):
        """显示错误消息框"""
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle(title)
        msg_box.exec_()

    def start_camera(self):
        if self.cap is None:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self._show_error_message("无法打开摄像头，请检查设备连接。", "摄像头错误")
                    return
                self.RecognizeButton.show()
                self.update_camera_frame()
            except Exception as e:
                self._show_error_message(f"打开摄像头时出错: {e}", "摄像头错误")

    def update_camera_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # 将OpenCV图像转换为Qt图像
                try:
                    height, width, channel = frame.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(qImg)
                    pixmap = pixmap.scaled(self.image_display_label.size(), aspectRatioMode=Qt.KeepAspectRatioByExpanding, transformMode=Qt.SmoothTransformation)
                    # 在QLabel中显示图像
                    self.image_display_label.setPixmap(pixmap)
                except Exception as e:
                    print(f"更新摄像头帧时出错: {e}")
            QtWidgets.QApplication.processEvents()
            self.image_display_label.window().timer = self.image_display_label.window().startTimer(30)

    def timerEvent(self, event):
        self.update_camera_frame()

    def recognize(self):
        if self.cap is not None:
            # 模拟识别结果，添加数据到表格
            self.add_data_to_table("实时摄像机", "未标记", "男", "25")            
            # 释放不必要的内存
            self._release_unused_memory()

    def stop_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
                cv2.destroyAllWindows()
                # 清空QLabel
                self.image_display_label.clear()
                # 隐藏识别按钮
                self.RecognizeButton.hide()
                try:
                    self.image_display_label.window().killTimer(self.image_display_label.window().timer)
                except AttributeError:
                    pass
            except Exception as e:
                print(f"停止摄像头时出错: {e}")

    def display_image(self, file_path_or_image):
        try:
            # 读取图片
            if isinstance(file_path_or_image, str):
                # 如果传入的是文件路径，读取图片
                image = cv2.imread(file_path_or_image)
            else:
                # 如果传入的是图像数组，直接使用
                image = file_path_or_image

            if image is None:
                self._show_error_message(f"无法读取图片文件: {file_path_or_image}", "图片读取错误")
                return
            
            # 获取图片的原始高度和宽度
            height, width = image.shape[:2]
            label_width = self.image_display_label.width()
            label_height = self.image_display_label.height()

            # 计算缩放比例
            if width / height > label_width / label_height:
                # 图片较宽，以宽度为基准缩放
                new_width = label_width
                new_height = int(height * (label_width / width))
            else:
                # 图片较高，以高度为基准缩放
                new_height = label_height
                new_width = int(width * (label_height / height))

            # 缩放图片
            resized_image = cv2.resize(image, (new_width, new_height))

            # 判断图片大小是否与QLabel大小一致
            if new_width == label_width and new_height == label_height:
                # 大小一致，不做修改
                pass
            else:
                # 大小不一致，取消原本QLabel的框
                self.image_display_label.setStyleSheet("")
                # 在缩放后的图片四周画框
                cv2.rectangle(resized_image, (0, 0), (new_width - 1, new_height - 1), (0, 0, 255), 2)

            # 将OpenCV图像转换为Qt图像
            bytesPerLine = 3 * new_width
            qImg = QImage(resized_image.data, new_width, new_height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            # 在QLabel中显示图像
            self.image_display_label.setPixmap(pixmap)
        except Exception as e:
            self._show_error_message(f"显示图像时出错: {e}", "图像显示错误")

    def add_data_to_table(self, location, tag, gender, age):
        """添加一行四列数据"""
        row = [
            QStandardItem(location),
            QStandardItem(tag),
            QStandardItem(gender),
            QStandardItem(age)
        ]
        self.model.appendRow(row)

    def _release_unused_memory(self):
        import gc
        gc.collect()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())