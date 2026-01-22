from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def isocurve(data, level, connected=False, extendToEdge=False, path=False):
    """
    Generate isocurve from 2D data using marching squares algorithm.
    
    ============== =========================================================
    **Arguments:**
    data           2D numpy array of scalar values
    level          The level at which to generate an isosurface
    connected      If False, return a single long list of point pairs
                   If True, return multiple long lists of connected point 
                   locations. (This is slower but better for drawing 
                   continuous lines)
    extendToEdge   If True, extend the curves to reach the exact edges of 
                   the data. 
    path           if True, return a QPainterPath rather than a list of 
                   vertex coordinates. This forces connected=True.
    ============== =========================================================
    
    This function is SLOW; plenty of room for optimization here.
    """
    if path is True:
        connected = True
    if extendToEdge:
        d2 = np.empty((data.shape[0] + 2, data.shape[1] + 2), dtype=data.dtype)
        d2[1:-1, 1:-1] = data
        d2[0, 1:-1] = data[0]
        d2[-1, 1:-1] = data[-1]
        d2[1:-1, 0] = data[:, 0]
        d2[1:-1, -1] = data[:, -1]
        d2[0, 0] = d2[0, 1]
        d2[0, -1] = d2[1, -1]
        d2[-1, 0] = d2[-1, 1]
        d2[-1, -1] = d2[-1, -2]
        data = d2
    sideTable = [[], [0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [0, 1, 2, 3], [2, 3], [2, 3], [0, 1, 2, 3], [1, 3], [0, 3], [0, 2], [1, 2], [0, 1], []]
    edgeKey = [[(0, 1), (0, 0)], [(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 1)]]
    lines = []
    mask = data < level
    index = np.zeros([x - 1 for x in data.shape], dtype=np.ubyte)
    fields = np.empty((2, 2), dtype=object)
    slices = [slice(0, -1), slice(1, None)]
    for i in [0, 1]:
        for j in [0, 1]:
            fields[i, j] = mask[slices[i], slices[j]]
            vertIndex = i + 2 * j
            np.add(index, fields[i, j] * 2 ** vertIndex, out=index, casting='unsafe')
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            sides = sideTable[index[i, j]]
            for l in range(0, len(sides), 2):
                edges = sides[l:l + 2]
                pts = []
                for m in [0, 1]:
                    p1 = edgeKey[edges[m]][0]
                    p2 = edgeKey[edges[m]][1]
                    v1 = data[i + p1[0], j + p1[1]]
                    v2 = data[i + p2[0], j + p2[1]]
                    f = (level - v1) / (v2 - v1)
                    fi = 1.0 - f
                    p = (p1[0] * fi + p2[0] * f + i + 0.5, p1[1] * fi + p2[1] * f + j + 0.5)
                    if extendToEdge:
                        p = (min(data.shape[0] - 2, max(0, p[0] - 1)), min(data.shape[1] - 2, max(0, p[1] - 1)))
                    if connected:
                        gridKey = (i + (1 if edges[m] == 2 else 0), j + (1 if edges[m] == 3 else 0), edges[m] % 2)
                        pts.append((p, gridKey))
                    else:
                        pts.append(p)
                lines.append(pts)
    if not connected:
        return lines
    points = {}
    for a, b in lines:
        if a[1] not in points:
            points[a[1]] = []
        points[a[1]].append([a, b])
        if b[1] not in points:
            points[b[1]] = []
        points[b[1]].append([b, a])
    for k in list(points.keys()):
        try:
            chains = points[k]
        except KeyError:
            continue
        for chain in chains:
            x = None
            while True:
                if x == chain[-1][1]:
                    break
                x = chain[-1][1]
                if x == k:
                    break
                y = chain[-2][1]
                connects = points[x]
                for conn in connects[:]:
                    if conn[1][1] != y:
                        chain.extend(conn[1:])
                del points[x]
            if chain[0][1] == chain[-1][1]:
                chains.pop()
                break
    lines = []
    for chain in points.values():
        if len(chain) == 2:
            chain = chain[1][1:][::-1] + chain[0]
        else:
            chain = chain[0]
        lines.append([p[0] for p in chain])
    if not path:
        return lines
    path = QtGui.QPainterPath()
    for line in lines:
        path.moveTo(*line[0])
        for p in line[1:]:
            path.lineTo(*p)
    return path