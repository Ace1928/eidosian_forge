import sys
from PySide2 import QtCore
from PySide2.QtCore import QDir, QFileInfo, QStandardPaths, Qt, QUrl
from PySide2.QtGui import QDesktopServices
from PySide2.QtWidgets import (QAction, QLabel, QMenu, QProgressBar,
from PySide2.QtWebEngineWidgets import QWebEngineDownloadItem
def mouse_double_click_event(self, event):
    if self.state() == QWebEngineDownloadItem.DownloadCompleted:
        self._launch()