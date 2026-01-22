from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def getImageFormatClass(self, imageFormat):
    return ebdt_bitmap_classes[imageFormat]