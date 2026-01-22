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
def getCoordinates(self, glyfTable):
    """Return the coordinates, end points and flags

        This method returns three values: A :py:class:`GlyphCoordinates` object,
        a list of the indexes of the final points of each contour (allowing you
        to split up the coordinates list into contours) and a list of flags.

        On simple glyphs, this method returns information from the glyph's own
        contours; on composite glyphs, it "flattens" all components recursively
        to return a list of coordinates representing all the components involved
        in the glyph.

        To interpret the flags for each point, see the "Simple Glyph Flags"
        section of the `glyf table specification <https://docs.microsoft.com/en-us/typography/opentype/spec/glyf#simple-glyph-description>`.
        """
    if self.numberOfContours > 0:
        return (self.coordinates, self.endPtsOfContours, self.flags)
    elif self.isComposite():
        allCoords = GlyphCoordinates()
        allFlags = bytearray()
        allEndPts = []
        for compo in self.components:
            g = glyfTable[compo.glyphName]
            try:
                coordinates, endPts, flags = g.getCoordinates(glyfTable)
            except RecursionError:
                raise ttLib.TTLibError("glyph '%s' contains a recursive component reference" % compo.glyphName)
            coordinates = GlyphCoordinates(coordinates)
            if hasattr(compo, 'firstPt'):
                if hasattr(compo, 'transform'):
                    coordinates.transform(compo.transform)
                x1, y1 = allCoords[compo.firstPt]
                x2, y2 = coordinates[compo.secondPt]
                move = (x1 - x2, y1 - y2)
                coordinates.translate(move)
            else:
                move = (compo.x, compo.y)
                if not hasattr(compo, 'transform'):
                    coordinates.translate(move)
                else:
                    apple_way = compo.flags & SCALED_COMPONENT_OFFSET
                    ms_way = compo.flags & UNSCALED_COMPONENT_OFFSET
                    assert not (apple_way and ms_way)
                    if not (apple_way or ms_way):
                        scale_component_offset = SCALE_COMPONENT_OFFSET_DEFAULT
                    else:
                        scale_component_offset = apple_way
                    if scale_component_offset:
                        coordinates.translate(move)
                        coordinates.transform(compo.transform)
                    else:
                        coordinates.transform(compo.transform)
                        coordinates.translate(move)
            offset = len(allCoords)
            allEndPts.extend((e + offset for e in endPts))
            allCoords.extend(coordinates)
            allFlags.extend(flags)
        return (allCoords, allEndPts, allFlags)
    elif self.isVarComposite():
        raise NotImplementedError('use TTGlyphSet to draw VarComposite glyphs')
    else:
        return (GlyphCoordinates(), [], bytearray())