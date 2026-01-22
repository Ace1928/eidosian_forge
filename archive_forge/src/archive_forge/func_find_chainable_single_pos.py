from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def find_chainable_single_pos(self, lookups, glyphs, value):
    """Helper for add_single_pos_chained_()"""
    res = None
    for lookup in lookups[::-1]:
        if lookup == self.SUBTABLE_BREAK_:
            return res
        if isinstance(lookup, SinglePosBuilder) and all((lookup.can_add(glyph, value) for glyph in glyphs)):
            res = lookup
    return res