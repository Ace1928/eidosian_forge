from fontTools.config import Config
from fontTools.misc import xmlWriter
from fontTools.misc.configTools import AbstractConfig
from fontTools.misc.textTools import Tag, byteord, tostr
from fontTools.misc.loggingTools import deprecateArgument
from fontTools.ttLib import TTLibError
from fontTools.ttLib.ttGlyphSet import _TTGlyph, _TTGlyphSetCFF, _TTGlyphSetGlyf
from fontTools.ttLib.sfnt import SFNTReader, SFNTWriter
from io import BytesIO, StringIO, UnsupportedOperation
import os
import logging
import traceback
def getGlyphOrder(self):
    """Returns a list of glyph names ordered by their position in the font."""
    try:
        return self.glyphOrder
    except AttributeError:
        pass
    if 'CFF ' in self:
        cff = self['CFF ']
        self.glyphOrder = cff.getGlyphOrder()
    elif 'post' in self:
        glyphOrder = self['post'].getGlyphOrder()
        if glyphOrder is None:
            self._getGlyphNamesFromCmap()
        elif len(glyphOrder) < self['maxp'].numGlyphs:
            log.warning("Not enough names found in the 'post' table, generating them from cmap instead")
            self._getGlyphNamesFromCmap()
        else:
            self.glyphOrder = glyphOrder
    else:
        self._getGlyphNamesFromCmap()
    return self.glyphOrder