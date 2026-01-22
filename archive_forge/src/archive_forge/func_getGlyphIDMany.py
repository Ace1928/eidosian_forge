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
def getGlyphIDMany(self, lst):
    """Converts a list of glyph names into a list of glyph IDs."""
    d = self.getReverseGlyphMap()
    try:
        return [d[glyphName] for glyphName in lst]
    except KeyError:
        getGlyphID = self.getGlyphID
        return [getGlyphID(glyphName) for glyphName in lst]