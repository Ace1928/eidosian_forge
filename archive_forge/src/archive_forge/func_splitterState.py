import inspect
import weakref
from .Qt import QtCore, QtWidgets
def splitterState(w):
    s = w.saveState().toPercentEncoding().data().decode()
    return s