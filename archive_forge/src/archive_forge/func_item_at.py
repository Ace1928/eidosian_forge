from PySide2.QtWebEngineWidgets import (QWebEnginePage, QWebEngineView,
from PySide2.QtWidgets import QApplication, QDesktopWidget, QTreeView
from PySide2.QtCore import (Signal, QAbstractTableModel, QModelIndex, Qt,
def item_at(self, model_index):
    return self._history.itemAt(model_index.row())