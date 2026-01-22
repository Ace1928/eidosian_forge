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
def toXML(self, writer, ttFont):
    glyphOrder = ttFont.getGlyphOrder()
    writer.comment("The 'id' attribute is only for humans; it is ignored when parsed.")
    writer.newline()
    for i in range(len(glyphOrder)):
        glyphName = glyphOrder[i]
        writer.simpletag('GlyphID', id=i, name=glyphName)
        writer.newline()