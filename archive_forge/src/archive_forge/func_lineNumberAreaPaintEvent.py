from PySide2.QtCore import Slot, Qt, QRect, QSize
from PySide2.QtGui import QColor, QPainter, QTextFormat
from PySide2.QtWidgets import QPlainTextEdit, QWidget, QTextEdit
def lineNumberAreaPaintEvent(self, event):
    painter = QPainter(self.line_number_area)
    painter.fillRect(event.rect(), Qt.lightGray)
    block = self.firstVisibleBlock()
    block_number = block.blockNumber()
    offset = self.contentOffset()
    top = self.blockBoundingGeometry(block).translated(offset).top()
    bottom = top + self.blockBoundingRect(block).height()
    while block.isValid() and top <= event.rect().bottom():
        if block.isVisible() and bottom >= event.rect().top():
            number = str(block_number + 1)
            painter.setPen(Qt.black)
            width = self.line_number_area.width()
            height = self.fontMetrics().height()
            painter.drawText(0, top, width, height, Qt.AlignRight, number)
        block = block.next()
        top = bottom
        bottom = top + self.blockBoundingRect(block).height()
        block_number += 1