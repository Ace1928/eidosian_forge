from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
@Slot()
def turnOff(self):
    self.setOn(False)