import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
def tabClicked(self, tab, ev=None):
    if ev is None or ev.button() == QtCore.Qt.MouseButton.LeftButton:
        for i in range(self.count()):
            w = self.widget(i)
            if w is tab.dock:
                w.label.setDim(False)
                self.stack.setCurrentIndex(i)
            else:
                w.label.setDim(True)