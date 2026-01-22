import os
import re
from ..GraphicsScene import GraphicsScene
from ..Qt import QtCore, QtWidgets
from ..widgets.FileDialog import FileDialog
def fileSaveFinished(self, fileName):
    global LastExportDirectory
    LastExportDirectory = os.path.split(fileName)[0]
    ext = os.path.splitext(fileName)[1].lower().lstrip('.')
    selectedExt = re.search('\\*\\.(\\w+)\\b', self.fileDialog.selectedNameFilter())
    if selectedExt is not None:
        selectedExt = selectedExt.groups()[0].lower()
        if ext != selectedExt:
            fileName = fileName + '.' + selectedExt.lstrip('.')
    self.export(fileName=fileName, **self.fileDialog.opts)