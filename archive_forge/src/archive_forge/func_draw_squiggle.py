import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def draw_squiggle(self, p1: point_like, p2: point_like, breadth=2):
    """Draw a squiggly line from p1 to p2."""
    p1 = Point(p1)
    p2 = Point(p2)
    S = p2 - p1
    rad = abs(S)
    cnt = 4 * int(round(rad / (4 * breadth), 0))
    if cnt < 4:
        raise ValueError('points too close')
    mb = rad / cnt
    matrix = Matrix(TOOLS._hor_matrix(p1, p2))
    i_mat = ~matrix
    k = 2.4142135623765633
    points = []
    for i in range(1, cnt):
        if i % 4 == 1:
            p = Point(i, -k) * mb
        elif i % 4 == 3:
            p = Point(i, k) * mb
        else:
            p = Point(i, 0) * mb
        points.append(p * i_mat)
    points = [p1] + points + [p2]
    cnt = len(points)
    i = 0
    while i + 2 < cnt:
        self.draw_curve(points[i], points[i + 1], points[i + 2])
        i += 2
    return p2