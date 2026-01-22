from PySide2.QtCore import Qt, QSize
from PySide2.QtWidgets import (QApplication, QDialog, QLayout, QGridLayout,
def rotateWidgets(self):
    count = len(self.rotableWidgets)
    if count % 2 == 1:
        raise AssertionError('Number of widgets must be even')
    for widget in self.rotableWidgets:
        self.rotableLayout.removeWidget(widget)
    self.rotableWidgets.append(self.rotableWidgets.pop(0))
    for i in range(count // 2):
        self.rotableLayout.addWidget(self.rotableWidgets[count - i - 1], 0, i)
        self.rotableLayout.addWidget(self.rotableWidgets[i], 1, i)