import os
from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import recursiveImport, strTypes
from reportlab.platypus import Frame
from reportlab.platypus import Flowable
from reportlab.platypus import Paragraph
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib.validators import isColor
from reportlab.lib.colors import toColor
from reportlab.lib.styles import _baseFontName, _baseFontNameI
def demo1(canvas):
    frame = Frame(2 * inch, 4 * inch, 4 * inch, 5 * inch, showBoundary=1)
    bodyStyle = ParagraphStyle('Body', fontName=_baseFontName, fontSize=24, leading=28, spaceBefore=6)
    para1 = Paragraph('Spam spam spam spam. ' * 5, bodyStyle)
    para2 = Paragraph('Eggs eggs eggs. ' * 5, bodyStyle)
    mydata = [para1, para2]
    frame.addFromList(mydata, canvas)