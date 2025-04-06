import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QMenuBar, QMenu, QAction, QLabel, QProgressBar, QTableWidget, QHeaderView,
    QStatusBar, QFrame, QGridLayout 
)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setWindowTitle("人脸识别系统")
        self.resize(1200, 800)

    def setup_ui(self):
        central_widget = QWidget(self)
        main_layout = QVBoxLayout(central_widget)

        # 顶部菜单栏
        self.setup_menu_bar()

        # 主内容区域
        content_layout = QHBoxLayout()
        self.setup_left_panel(content_layout)
        self.setup_right_panel(content_layout)
        main_layout.addLayout(content_layout)

        # 历史记录
        self.setup_history_table(main_layout)

        # 状态栏
        self.setup_status_bar()

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 样式设置
        self.setStyleSheet("""
            QWidget {
                font-family: 'Open Sans';
                font-size: 14px;
            }
            QMenuBar {
                background-color: #611bf8;
                color: white;
                padding: 4px;
            }
            QMenuBar::item {
                padding: 4px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:hover {
                background-color: #631bff;
            }
            QMenu {
                background-color: white;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 32px;
            }
            QMenu::item:hover {
                background-color: #f3f4f6;
            }
            QProgressBar {
                height: 16px;
                background: #e5e7eb;
                border-radius: 8px;
            }
            QProgressBar::chunk {
                background-color: #7311ff;
                border-radius: 8px;
            }
        """)

    def setup_menu_bar(self):
        menu_bar = QMenuBar(self)

        # 文件菜单
        file_menu = QMenu("文件", self)
        file_menu.addAction(QIcon(":/icons/image.svg"), "打开图片")
        file_menu.addAction(QIcon(":/icons/movie.svg"), "打开视频")
        file_menu.addSeparator()
        file_menu.addAction(QIcon(":/icons/save.svg"), "保存识别结果")
        file_menu.addSeparator()
        file_menu.addAction(QIcon(":/icons/logout.svg"), "退出")

        # 操作菜单
        action_menu = QMenu("操作", self)
        action_menu.addAction(QIcon(":/icons/play.svg"), "开始识别")
        action_menu.addAction(QIcon(":/icons/pause.svg"), "暂停识别")
        action_menu.addAction(QIcon(":/icons/resume.svg"), "继续识别")
        action_menu.addAction(QIcon(":/icons/stop.svg"), "停止识别")

        # 设置菜单
        settings_menu = QMenu("设置", self)
        settings_menu.addAction(QIcon(":/icons/settings.svg"), "系统设置")
        settings_menu.addAction(QIcon(":/icons/tune.svg"), "识别参数设置")

        # 帮助菜单
        help_menu = QMenu("帮助", self)
        help_menu.addAction(QIcon(":/icons/help.svg"), "使用说明")
        help_menu.addAction(QIcon(":/icons/info.svg"), "关于")
        help_menu.addAction(QIcon(":/icons/update.svg"), "检查更新")

        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(action_menu)
        menu_bar.addMenu(settings_menu)
        menu_bar.addMenu(help_menu)

        # 登录按钮
        login_btn = QPushButton("登录")
        login_btn.setIcon(QIcon(":/icons/login.svg"))
        menu_bar.setCornerWidget(login_btn, Qt.TopRightCorner)

        self.setMenuBar(menu_bar)

    def setup_left_panel(self, parent):
        left_panel = QWidget()
        layout = QVBoxLayout(left_panel)

        # 图像显示区域
        grid = QGridLayout()
        self.add_image_panel(grid, 0, "原始图像显示区", "点击打开图片或视频加载文件")
        self.add_image_panel(grid, 1, "处理后图像显示区", "完成识别后显示标记的人脸信息")
        layout.addLayout(grid)

        # 控制按钮
        btn_layout = QHBoxLayout()
        start_btn = QPushButton("开始识别")
        start_btn.setIcon(QIcon(":/icons/play.svg"))
        pause_btn = QPushButton("暂停识别")
        pause_btn.setEnabled(False)
        stop_btn = QPushButton("停止识别")
        stop_btn.setEnabled(False)

        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(pause_btn)
        btn_layout.addWidget(stop_btn)
        layout.addLayout(btn_layout)

        # 进度条
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        layout.addWidget(progress_bar)

        parent.addWidget(left_panel, 2)

    def add_image_panel(self, grid, col, title, hint):
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setStyleSheet("border: 2px dashed #d1d5db; border-radius: 12px; background: #f9fafb;")

        layout = QVBoxLayout(frame)
        icon = QLabel()
        try:
            pixmap = QPixmap(":/icons/image.svg").scaled(64, 64)
            icon.setPixmap(pixmap)
        except Exception as e:
            print(f"图标加载失败: {e}")
        layout.addWidget(icon, 0, Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #6b7280; margin-top: 8px;")
        layout.addWidget(title_label, 0, Qt.AlignCenter)

        hint_label = QLabel(hint)
        hint_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        layout.addWidget(hint_label, 0, Qt.AlignCenter)

        grid.addWidget(frame, 0, col)

    def setup_right_panel(self, parent):
        right_panel = QWidget()
        layout = QVBoxLayout(right_panel)

        # 识别结果
        result_frame = QFrame()
        result_frame.setStyleSheet("background: #f9fafb; border-radius: 12px; padding: 12px;")

        result_label = QLabel("尚无识别结果")
        result_label.setStyleSheet("color: #6b7280; font-style: italic;")
        result_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(result_frame)
        parent.addWidget(right_panel, 1)

    def setup_history_table(self, parent):
        table = QTableWidget(5, 5)
        table.setHorizontalHeaderLabels(["时间", "文件名", "人脸数量", "识别结果", "操作"])
        table.horizontalHeader().setStyleSheet("color: #6b7280; font-weight: bold;")
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        parent.addWidget(table)

    def setup_status_bar(self):
        self.statusBar().showMessage("状态: 就绪")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    