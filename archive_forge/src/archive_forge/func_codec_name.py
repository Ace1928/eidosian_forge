from PySide2 import QtCore, QtGui, QtWidgets
def codec_name(codec):
    try:
        name = str(codec.name(), encoding='ascii')
    except TypeError:
        name = str(codec.name())
    return name