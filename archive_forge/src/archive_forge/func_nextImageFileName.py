import os, sys
from PySide2.QtCore import QDate, QDir, QStandardPaths, Qt, QUrl
from PySide2.QtGui import QClipboard, QGuiApplication, QDesktopServices, QIcon
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import (QAction, qApp, QApplication, QHBoxLayout, QLabel,
from PySide2.QtMultimedia import QCamera, QCameraImageCapture, QCameraInfo
from PySide2.QtMultimediaWidgets import QCameraViewfinder
def nextImageFileName(self):
    picturesLocation = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
    dateString = QDate.currentDate().toString('yyyyMMdd')
    pattern = picturesLocation + '/pyside2_camera_' + dateString + '_{:03d}.jpg'
    n = 1
    while True:
        result = pattern.format(n)
        if not os.path.exists(result):
            return result
        n = n + 1
    return None