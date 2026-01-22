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
def decompileCoordinates(self, data):
    endPtsOfContours = array.array('H')
    endPtsOfContours.frombytes(data[:2 * self.numberOfContours])
    if sys.byteorder != 'big':
        endPtsOfContours.byteswap()
    self.endPtsOfContours = endPtsOfContours.tolist()
    pos = 2 * self.numberOfContours
    instructionLength, = struct.unpack('>h', data[pos:pos + 2])
    self.program = ttProgram.Program()
    self.program.fromBytecode(data[pos + 2:pos + 2 + instructionLength])
    pos += 2 + instructionLength
    nCoordinates = self.endPtsOfContours[-1] + 1
    flags, xCoordinates, yCoordinates = self.decompileCoordinatesRaw(nCoordinates, data, pos)
    self.coordinates = coordinates = GlyphCoordinates.zeros(nCoordinates)
    xIndex = 0
    yIndex = 0
    for i in range(nCoordinates):
        flag = flags[i]
        if flag & flagXShort:
            if flag & flagXsame:
                x = xCoordinates[xIndex]
            else:
                x = -xCoordinates[xIndex]
            xIndex = xIndex + 1
        elif flag & flagXsame:
            x = 0
        else:
            x = xCoordinates[xIndex]
            xIndex = xIndex + 1
        if flag & flagYShort:
            if flag & flagYsame:
                y = yCoordinates[yIndex]
            else:
                y = -yCoordinates[yIndex]
            yIndex = yIndex + 1
        elif flag & flagYsame:
            y = 0
        else:
            y = yCoordinates[yIndex]
            yIndex = yIndex + 1
        coordinates[i] = (x, y)
    assert xIndex == len(xCoordinates)
    assert yIndex == len(yCoordinates)
    coordinates.relativeToAbsolute()
    for i in range(len(flags)):
        flags[i] &= keepFlags
    self.flags = flags