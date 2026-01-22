from PySide2 import QtCore, QtGui, QtWidgets, QtXmlPatterns
import schema_rc
from ui_schema import Ui_SchemaMainWindow
def handleMessage(self, type, description, identifier, sourceLocation):
    self.m_description = description
    self.m_sourceLocation = sourceLocation