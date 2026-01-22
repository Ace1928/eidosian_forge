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
def getDrawing04():
    """Text strings in various colours.

    Colours are blue, yellow and red from bottom left
    to upper right.
    """
    D = Drawing(400, 200)
    i = 0
    for color in (colors.blue, colors.yellow, colors.red):
        D.add(String(50 + i * 30, 50 + i * 30, 'Hello World', fillColor=color))
        i = i + 1
    return D