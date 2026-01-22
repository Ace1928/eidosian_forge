from fontTools import config
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables
from fontTools.ttLib.tables.otBase import USE_HARFBUZZ_REPACKER
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.pens.basePen import NullPen
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.subset.util import _add_method, _uniq_sort
from fontTools.subset.cff import *
from fontTools.subset.svg import *
from fontTools.varLib import varStore  # for subset_varidxes
from fontTools.ttLib.tables._n_a_m_e import NameRecordVisitor
import sys
import struct
import array
import logging
from collections import Counter, defaultdict
from functools import reduce
from types import MethodType
@timer('load font')
def load_font(fontFile, options, checkChecksums=0, dontLoadGlyphNames=False, lazy=True):
    font = ttLib.TTFont(fontFile, checkChecksums=checkChecksums, recalcBBoxes=options.recalc_bounds, recalcTimestamp=options.recalc_timestamp, lazy=lazy, fontNumber=options.font_number)
    if dontLoadGlyphNames:
        post = ttLib.getTableClass('post')
        saved = post.decode_format_2_0
        post.decode_format_2_0 = post.decode_format_3_0
        f = font['post']
        if f.formatType == 2.0:
            f.formatType = 3.0
        post.decode_format_2_0 = saved
    return font