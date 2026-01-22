import logging
import sys
from .indenter import write_code
from .qtproxies import QtGui, QtWidgets, Literal, strict_getattr
def createQtGuiWidgetsWrappers(self):
    return [_QtGuiWrapper, _QtWidgetsWrapper]