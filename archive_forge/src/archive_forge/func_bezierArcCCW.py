from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
def bezierArcCCW(self, cx, cy, rx, ry, theta0, theta1):
    """return a set of control points for Bezier approximation to an arc
        with angle increasing counter clockwise. No requirement on (theta1-theta0) <= 90
        However, it must be true that theta1-theta0 > 0."""
    angularExtent = theta1 - theta0
    if abs(angularExtent) <= 90.0:
        angleList = [(theta0, theta1)]
    else:
        Nfrag = int(ceil(abs(angularExtent) / 90.0))
        fragAngle = float(angularExtent) / Nfrag
        angleList = []
        for ii in range(Nfrag):
            a = theta0 + ii * fragAngle
            b = a + fragAngle
            angleList.append((a, b))
    ctrlpts = []
    for a, b in angleList:
        if not ctrlpts:
            [(x0, y0), pts] = self._bezierArcSegmentCCW(cx, cy, rx, ry, a, b)
            ctrlpts.append(pts)
        else:
            [(tmpx, tmpy), pts] = self._bezierArcSegmentCCW(cx, cy, rx, ry, a, b)
            ctrlpts.append(pts)
    return ((x0, y0), ctrlpts)