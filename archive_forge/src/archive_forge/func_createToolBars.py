from PySide2 import QtCore, QtGui, QtWidgets
import application_rc
def createToolBars(self):
    self.fileToolBar = self.addToolBar('File')
    self.fileToolBar.addAction(self.newAct)
    self.fileToolBar.addAction(self.openAct)
    self.fileToolBar.addAction(self.saveAct)
    self.editToolBar = self.addToolBar('Edit')
    self.editToolBar.addAction(self.cutAct)
    self.editToolBar.addAction(self.copyAct)
    self.editToolBar.addAction(self.pasteAct)