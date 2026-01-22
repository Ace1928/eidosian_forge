import importlib.abc
import sys
import os
import types
from functools import partial, lru_cache
import operator
def import_pyqt6():
    """
    Import PyQt6

    ImportErrors raised within this function are non-recoverable
    """
    from PyQt6 import QtCore, QtSvg, QtWidgets, QtGui
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtGuiCompat = types.ModuleType('QtGuiCompat')
    QtGuiCompat.__dict__.update(QtGui.__dict__)
    QtGuiCompat.__dict__.update(QtWidgets.__dict__)
    api = QT_API_PYQT6
    return (QtCore, QtGuiCompat, QtSvg, api)