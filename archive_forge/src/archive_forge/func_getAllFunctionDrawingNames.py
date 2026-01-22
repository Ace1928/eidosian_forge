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
def getAllFunctionDrawingNames(doTTF=1):
    """Get a list of drawing function names from somewhere."""
    funcNames = []
    symbols = list(globals().keys())
    symbols.sort()
    for funcName in symbols:
        if funcName[0:10] == 'getDrawing':
            if doTTF or funcName != 'getDrawing13':
                funcNames.append(funcName)
    return funcNames