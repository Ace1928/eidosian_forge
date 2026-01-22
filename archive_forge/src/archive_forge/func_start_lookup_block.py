from fontTools.misc import sstruct
from fontTools.misc.textTools import Tag, tostr, binary2num, safeEval
from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lookupDebugInfo import (
from fontTools.feaLib.parser import Parser
from fontTools.feaLib.ast import FeatureFile
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.otlLib import builder as otl
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.ttLib import newTable, getTableModule
from fontTools.ttLib.tables import otBase, otTables
from fontTools.otlLib.builder import (
from fontTools.otlLib.error import OpenTypeLibError
from fontTools.varLib.varStore import OnlineVarStoreBuilder
from fontTools.varLib.builder import buildVarDevTable
from fontTools.varLib.featureVars import addFeatureVariationsRaw
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from collections import defaultdict
import copy
import itertools
from io import StringIO
import logging
import warnings
import os
def start_lookup_block(self, location, name):
    if name in self.named_lookups_:
        raise FeatureLibError('Lookup "%s" has already been defined' % name, location)
    if self.cur_feature_name_ == 'aalt':
        raise FeatureLibError("Lookup blocks cannot be placed inside 'aalt' features; move it out, and then refer to it with a lookup statement", location)
    self.cur_lookup_name_ = name
    self.named_lookups_[name] = None
    self.cur_lookup_ = None
    if self.cur_feature_name_ is None:
        self.lookupflag_ = 0
        self.lookupflag_markFilterSet_ = None