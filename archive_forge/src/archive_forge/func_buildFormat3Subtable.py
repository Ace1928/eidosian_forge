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
def buildFormat3Subtable(self, rule, chaining=True):
    st = self.newSubtable_(chaining=chaining)
    st.Format = 3
    if chaining:
        self.setBacktrackCoverage_(rule.prefix, st)
        self.setLookAheadCoverage_(rule.suffix, st)
        self.setInputCoverage_(rule.glyphs, st)
    else:
        self.setCoverage_(rule.glyphs, st)
    self.buildLookupList(rule, st)
    return st