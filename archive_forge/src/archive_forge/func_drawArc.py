import math
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import getFont, stringWidth, unicode2T1 # for font info
from reportlab.lib.utils import asBytes, char2int, rawBytes, asNative, isUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS
from reportlab import rl_config
from reportlab.pdfgen.canvas import FILL_EVEN_ODD
from reportlab.graphics.shapes import *
def drawArc(self, x1, y1, x2, y2, startAng=0, extent=360, fromcenter=0):
    """Draw a partial ellipse inscribed within the rectangle x1,y1,x2,y2,
        starting at startAng degrees and covering extent degrees.   Angles
        start with 0 to the right (+x) and increase counter-clockwise.
        These should have x1<x2 and y1<y2."""
    cx, cy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    rx, ry = ((x2 - x1) / 2.0, (y2 - y1) / 2.0)
    codeline = self._genArcCode(x1, y1, x2, y2, startAng, extent)
    startAngleRadians = math.pi * startAng / 180.0
    extentRadians = math.pi * extent / 180.0
    endAngleRadians = startAngleRadians + extentRadians
    codelineAppended = 0
    if self._fillColor != None:
        self.setColor(self._fillColor)
        self.code_append(codeline)
        codelineAppended = 1
        if self._strokeColor != None:
            self.code_append('gsave')
        self.lineTo(cx, cy)
        self.code_append('eofill')
        if self._strokeColor != None:
            self.code_append('grestore')
    if self._strokeColor != None:
        self.setColor(self._strokeColor)
        startx, starty = (cx + rx * math.cos(startAngleRadians), cy + ry * math.sin(startAngleRadians))
        if not codelineAppended:
            self.code_append(codeline)
        if fromcenter:
            self.lineTo(cx, cy)
            self.lineTo(startx, starty)
            self.code_append('closepath')
        self.code_append('stroke')