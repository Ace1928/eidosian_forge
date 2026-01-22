import sys
import re
from PySide2.QtCore import (QFile, Qt, QTextStream)
from PySide2.QtGui import (QColor, QFont, QKeySequence, QSyntaxHighlighter,
from PySide2.QtWidgets import (QAction, qApp, QApplication, QFileDialog, QMainWindow,
import syntaxhighlighter_rc
def setupEditor(self):
    variableFormat = QTextCharFormat()
    variableFormat.setFontWeight(QFont.Bold)
    variableFormat.setForeground(Qt.blue)
    self.highlighter.addMapping('\\b[A-Z_]+\\b', variableFormat)
    singleLineCommentFormat = QTextCharFormat()
    singleLineCommentFormat.setBackground(QColor('#77ff77'))
    self.highlighter.addMapping('#[^\n]*', singleLineCommentFormat)
    quotationFormat = QTextCharFormat()
    quotationFormat.setBackground(Qt.cyan)
    quotationFormat.setForeground(Qt.blue)
    self.highlighter.addMapping('".*"', quotationFormat)
    functionFormat = QTextCharFormat()
    functionFormat.setFontItalic(True)
    functionFormat.setForeground(Qt.blue)
    self.highlighter.addMapping('\\b[a-z0-9_]+\\(.*\\)', functionFormat)
    font = QFont()
    font.setFamily('Courier')
    font.setFixedPitch(True)
    font.setPointSize(10)
    self.editor = QPlainTextEdit()
    self.editor.setFont(font)
    self.highlighter.setDocument(self.editor.document())