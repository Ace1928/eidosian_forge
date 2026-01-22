from PySide2.QtCore import (QFile, QFileInfo, QPoint, QSettings, QSignalMapper,
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
import mdi_rc
def findMdiChild(self, fileName):
    canonicalFilePath = QFileInfo(fileName).canonicalFilePath()
    for window in self.mdiArea.subWindowList():
        if window.widget().currentFile() == canonicalFilePath:
            return window
    return None