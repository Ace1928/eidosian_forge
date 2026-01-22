from PySide2.QtCore import Slot, Qt, QRect, QSize
from PySide2.QtGui import QColor, QPainter, QTextFormat
from PySide2.QtWidgets import QPlainTextEdit, QWidget, QTextEdit
@Slot()
def update_line_number_area_width(self, newBlockCount):
    self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)