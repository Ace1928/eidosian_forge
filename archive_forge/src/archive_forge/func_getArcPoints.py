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
def getArcPoints(centerx, centery, radius, startangledegrees, endangledegrees, yradius=None, degreedelta=None, reverse=None):
    if yradius is None:
        yradius = radius
    points = []
    degreestoradians = pi / 180.0
    startangle = startangledegrees * degreestoradians
    endangle = endangledegrees * degreestoradians
    while endangle < startangle:
        endangle = endangle + 2 * pi
    angle = float(endangle - startangle)
    a = points.append
    if angle > 0.001:
        degreedelta = min(angle, degreedelta or 1.0)
        radiansdelta = degreedelta * degreestoradians
        n = max(int(angle / radiansdelta + 0.5), 1)
        radiansdelta = angle / n
        n += 1
    else:
        n = 1
        radiansdelta = 0
    for angle in range(n):
        angle = startangle + angle * radiansdelta
        a((centerx + radius * cos(angle), centery + yradius * sin(angle)))
    if reverse:
        points.reverse()
    return points