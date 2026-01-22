import os
import re
import subprocess
import sys
import time
import warnings
from . import QtCore, QtGui, QtWidgets, compat
from . import internals
def mkQApp(name=None):
    """
    Creates new QApplication or returns current instance if existing.
    
    ============== ========================================================
    **Arguments:**
    name           (str) Application name, passed to Qt
    ============== ========================================================
    """
    global QAPP

    def onPaletteChange(palette):
        color = palette.base().color()
        app = QtWidgets.QApplication.instance()
        darkMode = color.lightnessF() < 0.5
        app.setProperty('darkMode', darkMode)
    QAPP = QtWidgets.QApplication.instance()
    if QAPP is None:
        qtVersionCompare = tuple(map(int, QtVersion.split('.')))
        if qtVersionCompare > (6, 0):
            pass
        elif qtVersionCompare > (5, 14):
            os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        else:
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
        QAPP = QtWidgets.QApplication(sys.argv or ['pyqtgraph'])
        QAPP.paletteChanged.connect(onPaletteChange)
        QAPP.paletteChanged.emit(QAPP.palette())
    if name is not None:
        QAPP.setApplicationName(name)
    return QAPP