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
def getPointsBounds(pointList):
    """Helper function for list of points"""
    first = pointList[0]
    if isSeq(first):
        xs = [xy[0] for xy in pointList]
        ys = [xy[1] for xy in pointList]
        return (min(xs), min(ys), max(xs), max(ys))
    else:
        return getPathBounds(pointList)