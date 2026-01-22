from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging
def decode_format_2_0(self, data, ttFont):
    numGlyphs, = struct.unpack('>H', data[:2])
    numGlyphs = int(numGlyphs)
    if numGlyphs > ttFont['maxp'].numGlyphs:
        numGlyphs = ttFont['maxp'].numGlyphs
    data = data[2:]
    indices = array.array('H')
    indices.frombytes(data[:2 * numGlyphs])
    if sys.byteorder != 'big':
        indices.byteswap()
    data = data[2 * numGlyphs:]
    maxIndex = max(indices)
    self.extraNames = extraNames = unpackPStrings(data, maxIndex - 257)
    self.glyphOrder = glyphOrder = [''] * int(ttFont['maxp'].numGlyphs)
    for glyphID in range(numGlyphs):
        index = indices[glyphID]
        if index > 257:
            try:
                name = extraNames[index - 258]
            except IndexError:
                name = ''
        else:
            name = standardGlyphOrder[index]
        glyphOrder[glyphID] = name
    self.build_psNameMapping(ttFont)