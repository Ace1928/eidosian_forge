from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
def convertCFFtoCFF2(varFont):
    cffTable = varFont['CFF ']
    lib_convertCFFToCFF2(cffTable.cff, varFont)
    newCFF2 = newTable('CFF2')
    newCFF2.cff = cffTable.cff
    varFont['CFF2'] = newCFF2
    del varFont['CFF ']