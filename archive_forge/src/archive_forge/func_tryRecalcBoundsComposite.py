from collections import namedtuple
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set
def tryRecalcBoundsComposite(self, glyfTable, *, boundsDone=None):
    """Try recalculating the bounds of a composite glyph that has
        certain constrained properties. Namely, none of the components
        have a transform other than an integer translate, and none
        uses the anchor points.

        Each glyph object stores its bounding box in the
        ``xMin``/``yMin``/``xMax``/``yMax`` attributes. These bounds must be
        recomputed when the ``coordinates`` change. The ``table__g_l_y_f`` bounds
        must be provided to resolve component bounds.

        Return True if bounds were calculated, False otherwise.
        """
    for compo in self.components:
        if hasattr(compo, 'firstPt') or hasattr(compo, 'transform'):
            return False
        if not float(compo.x).is_integer() or not float(compo.y).is_integer():
            return False
    bounds = None
    for compo in self.components:
        glyphName = compo.glyphName
        g = glyfTable[glyphName]
        if boundsDone is None or glyphName not in boundsDone:
            g.recalcBounds(glyfTable, boundsDone=boundsDone)
            if boundsDone is not None:
                boundsDone.add(glyphName)
        if g.numberOfContours == 0:
            continue
        x, y = (compo.x, compo.y)
        bounds = updateBounds(bounds, (g.xMin + x, g.yMin + y))
        bounds = updateBounds(bounds, (g.xMax + x, g.yMax + y))
    if bounds is None:
        bounds = (0, 0, 0, 0)
    self.xMin, self.yMin, self.xMax, self.yMax = bounds
    return True