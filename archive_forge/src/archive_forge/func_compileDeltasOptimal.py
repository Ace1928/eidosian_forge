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
def compileDeltasOptimal(self, flags, deltas):
    candidates = []
    bestTuple = None
    bestCost = 0
    repeat = 0
    for flag, (x, y) in zip(flags, deltas):
        flag, coordBytes = flagBest(x, y, flag)
        bestCost += 1 + coordBytes
        newCandidates = [(bestCost, bestTuple, flag, coordBytes), (bestCost + 1, bestTuple, flag | flagRepeat, coordBytes)]
        for lastCost, lastTuple, lastFlag, coordBytes in candidates:
            if lastCost + coordBytes <= bestCost + 1 and lastFlag & flagRepeat and (lastFlag < 65280) and flagSupports(lastFlag, flag):
                if lastFlag & 255 == flag | flagRepeat and lastCost == bestCost + 1:
                    continue
                newCandidates.append((lastCost + coordBytes, lastTuple, lastFlag + 256, coordBytes))
        candidates = newCandidates
        bestTuple = min(candidates, key=lambda t: t[0])
        bestCost = bestTuple[0]
    flags = []
    while bestTuple:
        cost, bestTuple, flag, coordBytes = bestTuple
        flags.append(flag)
    flags.reverse()
    compressedFlags = bytearray()
    compressedXs = bytearray()
    compressedYs = bytearray()
    coords = iter(deltas)
    ff = []
    for flag in flags:
        repeatCount, flag = (flag >> 8, flag & 255)
        compressedFlags.append(flag)
        if flag & flagRepeat:
            assert repeatCount > 0
            compressedFlags.append(repeatCount)
        else:
            assert repeatCount == 0
        for i in range(1 + repeatCount):
            x, y = next(coords)
            flagEncodeCoords(flag, x, y, compressedXs, compressedYs)
            ff.append(flag)
    try:
        next(coords)
        raise Exception('internal error')
    except StopIteration:
        pass
    return (compressedFlags, compressedXs, compressedYs)