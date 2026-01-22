import importlib.abc
import sys
import os
import types
from functools import partial, lru_cache
import operator
def import_pyside6():
    """
    Import PySide6

    ImportErrors raised within this function are non-recoverable
    """
    from PySide6 import QtGui, QtCore, QtSvg, QtWidgets, QtPrintSupport
    QtGuiCompat = types.ModuleType('QtGuiCompat')
    QtGuiCompat.__dict__.update(QtGui.__dict__)
    QtGuiCompat.__dict__.update(QtWidgets.__dict__)
    QtGuiCompat.__dict__.update(QtPrintSupport.__dict__)
    return (QtCore, QtGuiCompat, QtSvg, QT_API_PYSIDE6)