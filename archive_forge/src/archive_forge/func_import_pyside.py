import sys
from functools import partial
from pydev_ipython.version import check_version
def import_pyside():
    """
    Import PySide

    ImportErrors raised within this function are non-recoverable
    """
    from PySide import QtGui, QtCore, QtSvg
    return (QtCore, QtGui, QtSvg, QT_API_PYSIDE)