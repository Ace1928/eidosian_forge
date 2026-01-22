from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
from . import grUtils
from array import array
from functools import reduce
import struct, re, sys
def writesimple(tag, self, writer, *attrkeys):
    attrs = dict([(k, getattr(self, k)) for k in attrkeys])
    writer.simpletag(tag, **attrs)
    writer.newline()