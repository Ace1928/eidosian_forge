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
def set_script(self, location, script):
    if self.cur_feature_name_ in ('aalt', 'size'):
        raise FeatureLibError('Script statements are not allowed within "feature %s"' % self.cur_feature_name_, location)
    if self.cur_feature_name_ is None:
        raise FeatureLibError('Script statements are not allowed within standalone lookup blocks', location)
    if self.language_systems == {(script, 'dflt')}:
        return
    self.cur_lookup_ = None
    self.script_ = script
    self.lookupflag_ = 0
    self.lookupflag_markFilterSet_ = None
    self.set_language(location, 'dflt', include_default=True, required=False)