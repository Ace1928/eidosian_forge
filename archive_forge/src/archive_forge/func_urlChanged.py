import sys
from PySide2.QtCore import QUrl
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import (QApplication, QDesktopWidget, QLineEdit,
from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
def urlChanged(self, url):
    self.addressLineEdit.setText(url.toString())