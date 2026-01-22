import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
@property
def styles(self):
    app = QtWidgets.QApplication.instance()
    return DARK_STYLES if app.property('darkMode') else LIGHT_STYLES