import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def drawRoundRect(self, x1, y1, x2, y2, rx=8, ry=8, edgeColor=None, edgeWidth=None, fillColor=None, dash=None, **kwargs):
    """Draw a rounded rectangle between x1,y1, and x2,y2,         with corners inset as ellipses with x radius rx and y radius ry.         These should have x1<x2, y1<y2, rx>0, and ry>0."""
    x1, x2 = (min(x1, x2), max(x1, x2))
    y1, y2 = (min(y1, y2), max(y1, y2))
    ellipsePath = 'matrix currentmatrix %s %s neg translate %s %s scale 0 0 1 %s %s arc setmatrix'
    rrcode = ['newpath']
    rrcode.append(ellipsePath % (x1 + rx, y1 + ry, rx, ry, 90, 180))
    rrcode.append(ellipsePath % (x1 + rx, y2 - ry, rx, ry, 180, 270))
    rrcode.append(ellipsePath % (x2 - rx, y2 - ry, rx, ry, 270, 360))
    rrcode.append(ellipsePath % (x2 - rx, y1 + ry, rx, ry, 0, 90))
    rrcode.append('closepath')
    self._updateFillColor(fillColor)
    if self._currentColor != transparent:
        self.code.extend(rrcode)
        self.code.append('eofill')
    self._updateLineWidth(edgeWidth)
    self._updateLineColor(edgeColor)
    if self._currentColor != transparent:
        self.code.extend(rrcode)
        self.code.append('stroke')