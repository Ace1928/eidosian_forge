from PySide2.QtCore import Slot, Qt, QRect, QSize
from PySide2.QtGui import QColor, QPainter, QTextFormat
from PySide2.QtWidgets import QPlainTextEdit, QWidget, QTextEdit
def line_number_area_width(self):
    digits = 1
    max_num = max(1, self.blockCount())
    while max_num >= 10:
        max_num *= 0.1
        digits += 1
    space = 3 + self.fontMetrics().width('9') * digits
    return space