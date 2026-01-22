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
def drawPage(canvas, x, y, width, height):
    pth = canvas.beginPath()
    corner = 0.05 * width
    canvas.setFillColorRGB(0.5, 0.5, 0.5)
    canvas.rect(x + corner, y - corner, width, height, stroke=0, fill=1)
    canvas.setFillColorRGB(1, 1, 0.9)
    canvas.setLineWidth(0)
    canvas.rect(x, y, width, height, stroke=1, fill=1)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.setStrokeColorRGB(0, 0, 0)