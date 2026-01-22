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
def getDrawing03():
    """Text strings in various sizes and different fonts.

    Font size increases from 12 to 36 and from bottom left
    to upper right corner.  The first ones should be in
    Times-Roman.  Finally, a solitary Courier string at
    the top right corner.
    """
    D = Drawing(400, 200)
    for size in range(12, 36, 4):
        D.add(String(10 + size * 2, 10 + size * 2, 'Hello World', fontName=_FONTS[0], fontSize=size))
    D.add(String(150, 150, 'Hello World', fontName=_FONTS[1], fontSize=36))
    return D