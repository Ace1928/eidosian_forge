from ..Qt import QtCore, QtGui, QtWidgets
def focusNext(self, forward=True):
    """Give focus to the next (or previous) focusable item in the parameter tree"""
    self.treeWidget().focusNext(self, forward=forward)