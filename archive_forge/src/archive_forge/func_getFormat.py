from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def getFormat(self):
    return safeEval(self.__class__.__name__[len(_bitmapGlyphSubclassPrefix):])