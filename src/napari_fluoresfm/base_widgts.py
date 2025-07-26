import os

import torch
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class FileSelectWidget(QWidget):
    """
    Allow users to select the path of a file.
    """

    def __init__(self, label="file", title="Select a file"):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # set the margins to 0
        self.setLayout(layout)

        layout.addWidget(QLabel(label))

        self.path_edit = QLineEdit()
        layout.addWidget(self.path_edit)

        self.btn_browse = QPushButton("Choose")
        self.btn_browse.released.connect(self._on_browse)
        layout.addWidget(self.btn_browse)

        self.title = title

        self.set_enabled(True)

        self.init_directory = os.getcwd()  # set default directory

    def _on_browse(self):
        file = QFileDialog.getOpenFileName(
            self, self.title, self.init_directory, "*.*"
        )
        if file != "":
            self.path_edit.setText(file[0])

    def get_path(self):
        return self.path_edit.text()

    def set_enabled(self, enable):
        self.path_edit.setEnabled(enable)
        self.btn_browse.setEnabled(enable)


class DirectorySelectWidget(FileSelectWidget):
    """
    Allow users to select the path of a directory.
    """

    def __init__(self, label="Folder", title="Select a folder"):
        super().__init__(label, title)

    def _on_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, self.title, self.init_directory
        )
        if directory != "":
            self.path_edit.setText(directory)


class ComboSelectBox(QWidget):
    def __init__(self, label, options=None):
        super().__init__()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # set the margins to 0
        self.setLayout(layout)

        layout.addWidget(QLabel(label))

        self.combo_box = QComboBox()
        if options is not None:
            self.combo_box.addItems(options)
        layout.addWidget(self.combo_box)

    def get_value(self):
        return self.combo_box.currentText()


class DeviceBox(ComboSelectBox):
    def __init__(self, label=""):
        super().__init__(label=label)

        device_list = self.get_device_list()
        self.combo_box.addItems(device_list)

    def get_device_list(self):
        device_list = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_list.append(f"cuda:{i}")
        return device_list

    def get_value(self):
        return self.combo_box.currentText()


class TextBox(QWidget):
    def __init__(self, label, box_type="LE"):
        super().__init__()

        self.box_type = box_type

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # set the margins to 0
        self.setLayout(layout)

        label_box = QLabel(label)
        label_box.setWordWrap(True)
        layout.addWidget(label_box)
        if self.box_type == "LE":
            self.text_box = QLineEdit()
        elif self.box_type == "TE":
            self.text_box = QTextEdit()
        elif self.box_type == "PTE":
            self.text_box = QPlainTextEdit()
        else:
            raise ValueError(f"Unknown box_type: {self.box_type}")
        layout.addWidget(self.text_box)

    def get_text(self):
        if self.box_type == "LE":
            text = self.text_box.text()
        elif self.box_type == "TE" or self.box_type == "PTE":
            text = self.text_box.toPlainText()
        return text

    def set_text(self, text):
        self.text_box.setText(text)


class SpinBox(QSpinBox):
    def __init__(self, vmin, vmax, vinit, step=1):
        super().__init__()

        self.setMinimum(vmin)
        self.setMaximum(vmax)
        self.setValue(vinit)
        self.setSingleStep(step)


class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, vmin, vmax, vinit, decimals=8, step=0.01):
        super().__init__()

        self.setDecimals(decimals)
        self.setMinimum(vmin)
        self.setMaximum(vmax)
        self.setValue(vinit)
        self.setSingleStep(step)


class ParamsBox(QWidget):
    def __init__(
        self,
        label,
        spintype="spin",
        vmin=0,
        vmax=None,
        vinit=0,
        decimals=8,
        step=1,
    ):
        super().__init__()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # set the margins to 0
        self.setLayout(layout)

        layout.addWidget(QLabel(label))
        if spintype == "spin":
            self.value_box = SpinBox(vmin, vmax, vinit, step=step)
        elif spintype == "doublespin":
            self.value_box = DoubleSpinBox(
                vmin, vmax, vinit, decimals, step=step
            )
        else:
            raise ValueError(f"Unknown type: {spintype}")
        layout.addWidget(self.value_box)

    def get_value(self):
        return self.value_box.value()

    def set_value(self, value):
        self.value_box.setValue(value)

    def set_enabled(self, enable):
        self.value_box.setEnabled(enable)


class Observer(QObject):
    progress_signal = Signal(int)
    progrss_total_signal = Signal(int)
    notify_signal = Signal(str)

    def __init__(self):
        super().__init__()

    def progress(self, value):
        self.progress_signal.emit(value)

    def notify(self, message):
        self.notify_signal.emit(message)

    def prograss_total(self, value):
        self.progrss_total_signal.emit(value)


class WorkerBase(QObject):
    finish_signal = Signal()
    succeed_signal = Signal()

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer
        self.params_dict = None
        self.stop_flag = [False]

    def set_params(self, params_dict):
        self.params_dict = params_dict

    def run(self):
        print("worker run ...")

    def stop(self):
        self.stop_flag[0] = True
        self.finish_signal.emit()


class WidgetBase(QGroupBox):
    def __init__(self, logger=None, worker=None, title=None):
        super().__init__()
        if title is not None:
            self.setTitle(title)

        self.logger = logger
        self.params = {}

        self._thread = QThread()
        self._observer = Observer()
        self._worker = (
            worker if worker is not None else WorkerBase(self._observer)
        )

        self.run_btn = QPushButton("run")
        self.stop_btn = QPushButton("stop")
        self.progress_bar = QProgressBar()

    def connect(self):
        self.run_btn.clicked.connect(self._on_click_run)
        self.stop_btn.clicked.connect(self._on_click_stop)

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finish_signal.connect(self._thread.quit)
        self._worker.finish_signal.connect(self.enable_btn)

        self._observer.progress_signal.connect(self._on_progress)
        self._observer.notify_signal.connect(self.log)
        self._observer.progrss_total_signal.connect(self.set_progress_total)

    def _on_progress(self, value):
        self.progress_bar.setValue(value)

    def set_progress_total(self, value):
        self.progress_bar.setMaximum(value)

    def log(self, value):
        print(value)
        if self.logger is not None:
            self.logger.add_text(value)

    def get_params(self):
        return self.params

    def _on_click_run(self):
        if self._thread.isRunning():
            self.log("thread is running, please wait or stop.")
            return
        self.log("run...")
        self.progress_bar.setValue(0)
        self._worker.stop_flag[0] = False
        self.get_params()
        self._worker.set_params(self.params)
        self._thread.start()
        self.run_btn.setEnabled(False)

    def _on_click_stop(self):
        self.log("stop...")
        self._worker.stop()
        self.log("stop finished.")
        self.run_btn.setEnabled(True)

    def enable_btn(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
