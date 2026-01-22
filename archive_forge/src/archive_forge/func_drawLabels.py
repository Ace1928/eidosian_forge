import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
def drawLabels(self, topLeft=''):
    """Writes little labels in the top row and first column"""
    self.canv.setFillGray(0.8)
    self.canv.rect(0, self.ylist[-2], self.width, self.boxSize, fill=1, stroke=0)
    self.canv.rect(0, 0, self.boxSize, self.ylist[-2], fill=1, stroke=0)
    self.canv.setFillGray(0.0)
    self.canv.setFont('Helvetica-Oblique', 0.375 * self.boxSize)
    byt = 0
    for row in range(self.rows):
        if self.rowLabels:
            label = self.rowLabels[row]
        else:
            label = self.formatByte(row * self.charsPerRow)
        self.canv.drawCentredString(0.5 * self.boxSize, (self.rows - row - 0.75) * self.boxSize, label)
    for col in range(self.charsPerRow):
        self.canv.drawCentredString((col + 1.5) * self.boxSize, (self.rows + 0.25) * self.boxSize, self.formatByte(col))
    if topLeft:
        self.canv.setFont('Helvetica-BoldOblique', 0.5 * self.boxSize)
        self.canv.drawCentredString(0.5 * self.boxSize, (self.rows + 0.25) * self.boxSize, topLeft)