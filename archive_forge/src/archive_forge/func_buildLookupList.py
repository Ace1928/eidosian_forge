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
def buildLookupList(self, rule, st):
    for sequenceIndex, lookupList in enumerate(rule.lookups):
        if lookupList is not None:
            if not isinstance(lookupList, list):
                lookupList = [lookupList]
            for l in lookupList:
                if l.lookup_index is None:
                    if isinstance(self, ChainContextPosBuilder):
                        other = 'substitution'
                    else:
                        other = 'positioning'
                    raise OpenTypeLibError(f'Missing index of the specified lookup, might be a {other} lookup', self.location)
                rec = self.newLookupRecord_(st)
                rec.SequenceIndex = sequenceIndex
                rec.LookupListIndex = l.lookup_index