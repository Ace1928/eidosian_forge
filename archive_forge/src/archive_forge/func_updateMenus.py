from PySide2.QtCore import (QFile, QFileInfo, QPoint, QSettings, QSignalMapper,
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
import mdi_rc
def updateMenus(self):
    hasMdiChild = self.activeMdiChild() is not None
    self.saveAct.setEnabled(hasMdiChild)
    self.saveAsAct.setEnabled(hasMdiChild)
    self.pasteAct.setEnabled(hasMdiChild)
    self.closeAct.setEnabled(hasMdiChild)
    self.closeAllAct.setEnabled(hasMdiChild)
    self.tileAct.setEnabled(hasMdiChild)
    self.cascadeAct.setEnabled(hasMdiChild)
    self.nextAct.setEnabled(hasMdiChild)
    self.previousAct.setEnabled(hasMdiChild)
    self.separatorAct.setVisible(hasMdiChild)
    hasSelection = self.activeMdiChild() is not None and self.activeMdiChild().textCursor().hasSelection()
    self.cutAct.setEnabled(hasSelection)
    self.copyAct.setEnabled(hasSelection)