from functools import partial
import sys
from bookmarkwidget import BookmarkWidget
from webengineview import WebEngineView
from historywindow import HistoryWindow
from PySide2 import QtCore
from PySide2.QtCore import QPoint, Qt, QUrl
from PySide2.QtWidgets import (QAction, QMenu, QTabBar, QTabWidget)
from PySide2.QtWebEngineWidgets import (QWebEngineDownloadItem,
def show_history(self):
    index = self.currentIndex()
    if index >= 0:
        webengineview = self._webengineviews[index]
        history_window = self._history_windows.get(webengineview)
        if not history_window:
            history = webengineview.page().history()
            history_window = HistoryWindow(history, self)
            history_window.open_url.connect(self.load)
            history_window.setWindowFlags(history_window.windowFlags() | Qt.Window)
            history_window.setWindowTitle('History')
            self._history_windows[webengineview] = history_window
        else:
            history_window.refresh()
        history_window.show()
        history_window.raise_()