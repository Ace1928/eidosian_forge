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
def set_size_parameters(self, location, DesignSize, SubfamilyID, RangeStart, RangeEnd):
    if self.cur_feature_name_ != 'size':
        raise FeatureLibError('Parameters statements are not allowed within "feature %s"' % self.cur_feature_name_, location)
    self.size_parameters_ = [DesignSize, SubfamilyID, RangeStart, RangeEnd]
    for script, lang in self.language_systems:
        key = (script, lang, self.cur_feature_name_)
        self.features_.setdefault(key, [])