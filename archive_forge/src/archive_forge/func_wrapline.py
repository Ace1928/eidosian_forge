from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
from . import grUtils
from array import array
from functools import reduce
import struct, re, sys
def wrapline(writer, dat, length=80):
    currline = ''
    for d in dat:
        if len(currline) > length:
            writer.write(currline[:-1])
            writer.newline()
            currline = ''
        currline += d + ' '
    if len(currline):
        writer.write(currline[:-1])
        writer.newline()