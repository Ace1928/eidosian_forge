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
def attachSubtableWithCount_(self, st, subtable_name, count_name, existing=None, index=None, chaining=False):
    if chaining:
        subtable_name = 'Chain' + subtable_name
        count_name = 'Chain' + count_name
    if not hasattr(st, count_name):
        setattr(st, count_name, 0)
        setattr(st, subtable_name, [])
    if existing:
        new_subtable = existing
    else:
        new_subtable = getattr(ot, subtable_name)()
    setattr(st, count_name, getattr(st, count_name) + 1)
    if index:
        getattr(st, subtable_name).insert(index, new_subtable)
    else:
        getattr(st, subtable_name).append(new_subtable)
    return new_subtable