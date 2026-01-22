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
def rulesets(self):
    ruleset = [ChainContextualRuleset()]
    for rule in self.rules:
        if rule.is_subtable_break:
            ruleset.append(ChainContextualRuleset())
            continue
        ruleset[-1].addRule(rule)
    return [x for x in ruleset if len(x.rules) > 0]