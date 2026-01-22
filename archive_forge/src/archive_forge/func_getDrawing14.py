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
def getDrawing14():
    """test shapes.Image"""
    from reportlab.graphics.shapes import Image
    D = Drawing(400, 200)
    im0 = smallArrow()
    D.add(Image(x=0, y=0, width=None, height=None, path=im0))
    im1 = smallArrow()
    D.add(Image(x=400 - 20, y=200 - 14, width=20, height=14, path=im1))
    return D