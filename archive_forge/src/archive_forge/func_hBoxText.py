import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
def hBoxText(msg, canvas, x, y, fontName):
    """Helper for stringwidth tests on Asian fonts.

    Registers font if needed.  Then draws the string,
    and a box around it derived from the stringWidth function"""
    canvas.saveState()
    try:
        font = pdfmetrics.getFont(fontName)
    except KeyError:
        font = cidfonts.UnicodeCIDFont(fontName)
        pdfmetrics.registerFont(font)
    canvas.setFillGray(0.8)
    canvas.rect(x, y, pdfmetrics.stringWidth(msg, fontName, 16), 16, stroke=0, fill=1)
    canvas.setFillGray(0)
    canvas.setFont(fontName, 16, 16)
    canvas.drawString(x, y, msg)
    canvas.restoreState()