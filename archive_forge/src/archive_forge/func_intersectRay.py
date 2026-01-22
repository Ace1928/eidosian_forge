import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def intersectRay(self, ray):
    h = self.h2
    r = self.r
    p, dir = ray.currentState(relativeTo=self)
    p = p - Point(r, 0)
    if r == 0:
        if dir[0] == 0:
            y = 0
        else:
            y = p[1] - p[0] * dir[1] / dir[0]
        if abs(y) > h:
            return (None, None)
        else:
            return (Point(0, y), atan2(dir[1], dir[0]))
    else:
        dx = dir[0]
        dy = dir[1]
        dr = hypot(dx, dy)
        D = p[0] * (p[1] + dy) - (p[0] + dx) * p[1]
        idr2 = 1.0 / dr ** 2
        disc = r ** 2 * dr ** 2 - D ** 2
        if disc < 0:
            return (None, None)
        disc2 = disc ** 0.5
        if dy < 0:
            sgn = -1
        else:
            sgn = 1
        br = self.path.boundingRect()
        x1 = (D * dy + sgn * dx * disc2) * idr2
        y1 = (-D * dx + abs(dy) * disc2) * idr2
        if br.contains(x1 + r, y1):
            pt = Point(x1, y1)
        else:
            x2 = (D * dy - sgn * dx * disc2) * idr2
            y2 = (-D * dx - abs(dy) * disc2) * idr2
            pt = Point(x2, y2)
            if not br.contains(x2 + r, y2):
                return (None, None)
        norm = atan2(pt[1], pt[0])
        if r < 0:
            norm += np.pi
        dp = p - pt
        ang = atan2(dp[1], dp[0])
        return (pt + Point(r, 0), ang - norm)