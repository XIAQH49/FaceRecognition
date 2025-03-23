from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QImage, QPixmap
# 新增导入
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
        self.model.setHorizontalHeaderLabels(["位置", "性别", "年龄"])  # 设置表头

        # 添加示例数据
        data = [
            ["摄像头", "男", "25"],
            ["./images...", "女", "25"]
        ]
        for row_data in data:
            row = []
            for item in row_data:
                cell = QStandardItem(item)
                row.append(cell)
            self.model.appendRow(row)

        self.table_view.setModel(self.model)

        # 设置列宽
        self.table_view.setColumnWidth(0, 439)
        self.table_view.setColumnWidth(1, 170)
        self.table_view.setColumnWidth(2, 170)

        # 连接按钮的clicked信号到选择文件的槽函数
        self.SelectButton.clicked.connect(self.select_file)

        # 连接开始和结束按钮的clicked信号
        self.StartButton.clicked.connect(self.start_camera)
        self.StopButton.clicked.connect(self.stop_camera)

        # 存储选择的文件路径
        self.selected_file_path = None
        # 初始化摄像头对象
        self.cap = None
        # 获取用于显示图像的QLabel
        self.image_display_label = self.DisplayLabel

        # 设置QLabel的边框为红色
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
                msg_box = QtWidgets.QMessageBox()
                msg_box.setIcon(QtWidgets.QMessageBox.Critical)
                msg_box.setText(f"不允许选择 {file_extension} 类型的文件，请选择图片文件。")
                msg_box.setWindowTitle("文件类型错误")
                msg_box.exec_()
            else:
                print(f"选择的文件路径: {file_path}")
                self.selected_file_path = file_path
                # 显示选择的图片
                self.display_image(file_path)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            while True:
                ret, frame = self.cap.read()
                if ret:
                    # 将OpenCV图像转换为Qt图像
                    height, width, channel = frame.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(qImg)
                    # 修正此处，使用正确的Qt引用
                    pixmap = pixmap.scaled(self.image_display_label.size(), aspectRatioMode=Qt.KeepAspectRatioByExpanding, transformMode=Qt.SmoothTransformation)
                    # 在QLabel中显示图像
                    self.image_display_label.setPixmap(pixmap)
                    QtWidgets.QApplication.processEvents()
                if cv2.waitKey(1) & 0xFF == ord('q') or self.cap is None:
                    break
            cv2.destroyAllWindows()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()
            # 清空QLabel
            self.image_display_label.clear()

    def display_image(self, file_path):
        try:
            # 读取图片
            image = cv2.imread(file_path)
            # 将OpenCV图像转换为Qt图像
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            # 修正此处，使用正确的Qt引用
            pixmap = pixmap.scaled(self.image_display_label.size(), aspectRatioMode=Qt.KeepAspectRatioByExpanding, transformMode=Qt.SmoothTransformation)
            # 在QLabel中显示图像
            self.image_display_label.setPixmap(pixmap)
        except Exception as e:
            print(f"显示图像时出错: {e}")
    