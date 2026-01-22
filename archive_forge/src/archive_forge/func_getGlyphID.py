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
def getGlyphID(self, glyphName):
    """Returns the ID of the glyph with the given name."""
    try:
        return self.getReverseGlyphMap()[glyphName]
    except KeyError:
        if glyphName[:5] == 'glyph':
            try:
                return int(glyphName[5:])
            except (NameError, ValueError):
                raise KeyError(glyphName)
        raise