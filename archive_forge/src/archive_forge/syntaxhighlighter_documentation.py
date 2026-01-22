import sys
import re
from PySide2.QtCore import (QFile, Qt, QTextStream)
from PySide2.QtGui import (QColor, QFont, QKeySequence, QSyntaxHighlighter,
from PySide2.QtWidgets import (QAction, qApp, QApplication, QFileDialog, QMainWindow,
import syntaxhighlighter_rc
PySide2 port of the widgets/richtext/syntaxhighlighter example from Qt v5.x