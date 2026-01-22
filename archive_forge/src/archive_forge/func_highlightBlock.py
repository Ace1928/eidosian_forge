import sys
import re
from PySide2.QtCore import (QFile, Qt, QTextStream)
from PySide2.QtGui import (QColor, QFont, QKeySequence, QSyntaxHighlighter,
from PySide2.QtWidgets import (QAction, qApp, QApplication, QFileDialog, QMainWindow,
import syntaxhighlighter_rc
def highlightBlock(self, text):
    for pattern in self.mappings:
        for m in re.finditer(pattern, text):
            s, e = m.span()
            self.setFormat(s, e - s, self.mappings[pattern])