import os, sys
from math import pi, cos, sin, sqrt, radians, floor
from reportlab.platypus import Flowable
from reportlab.rl_config import shapeChecking, verbose, defaultGraphicsFontName as _baseGFontName, _unset_, decimalSymbol
from reportlab.lib import logger
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.utils import isSeq, asBytes
from reportlab.lib.attrmap import *
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.fonts import tt2ps
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from . transform import *
def getBounds(self):
    t = self.text
    w = stringWidth(t, self.fontName, self.fontSize, self.encoding)
    tA = self.textAnchor
    x = self.x
    if tA != 'start':
        if tA == 'middle':
            x -= 0.5 * w
        elif tA == 'end':
            x -= w
        elif tA == 'numeric':
            x -= numericXShift(tA, t, w, self.fontName, self.fontSize, self.encoding)
    return (x, self.y - 0.2 * self.fontSize, x + w, self.y + self.fontSize)