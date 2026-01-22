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
def find_chainable_single_subst(self, mapping):
    """Helper for add_single_subst_chained_()"""
    res = None
    for rule in self.rules[::-1]:
        if rule.is_subtable_break:
            return res
        for sub in rule.lookups:
            if isinstance(sub, SingleSubstBuilder) and (not any((g in mapping and mapping[g] != sub.mapping[g] for g in sub.mapping))):
                res = sub
    return res