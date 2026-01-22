from PySide2 import QtCore, QtGui, QtWidgets, QtXmlPatterns
import schema_rc
from ui_schema import Ui_SchemaMainWindow
def encode_utf8(ba):
    return str(ba.data(), encoding='utf8')