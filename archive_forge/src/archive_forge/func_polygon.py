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
def polygon(self, p, closed=0, stroke=1, fill=1):
    assert len(p) >= 2, 'Polygon must have 2 or more points'
    start = p[0]
    p = p[1:]
    poly = []
    a = poly.append
    a('%s m' % fp_str(start))
    for point in p:
        a('%s l' % fp_str(point))
    if closed:
        a('closepath')
    self._fillAndStroke(poly, stroke=stroke, fill=fill)