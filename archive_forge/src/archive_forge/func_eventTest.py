from PySide2.QtWidgets import *
from PySide2.QtCore import *
def eventTest(self, e):
    return e.type() == QEvent.User + 2