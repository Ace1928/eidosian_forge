import os, sys
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import asNative, base64_decodebytes
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import *
import unittest
from reportlab.rl_config import register_reset
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
def getDrawing05():
    """Text strings with various anchors (alignments).

    Text alignment conforms to the anchors in the left column.
    """
    D = Drawing(400, 200)
    lineX = 250
    D.add(Line(lineX, 10, lineX, 190, strokeColor=colors.gray))
    y = 130
    for anchor in ('start', 'middle', 'end'):
        D.add(String(lineX, y, 'Hello World', textAnchor=anchor))
        D.add(String(50, y, anchor + ':'))
        y = y - 30
    return D