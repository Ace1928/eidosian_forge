from collections import UserDict, deque
from functools import partial
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
import array
import itertools
import logging
import struct
import sys
import fontTools.ttLib.tables.TupleVariation as tv
@staticmethod
def getNumPoints_(glyph):
    NUM_PHANTOM_POINTS = 4
    if glyph.isComposite():
        return len(glyph.components) + NUM_PHANTOM_POINTS
    elif glyph.isVarComposite():
        count = 0
        for component in glyph.components:
            count += component.getPointCount()
        return count + NUM_PHANTOM_POINTS
    else:
        return len(getattr(glyph, 'coordinates', [])) + NUM_PHANTOM_POINTS