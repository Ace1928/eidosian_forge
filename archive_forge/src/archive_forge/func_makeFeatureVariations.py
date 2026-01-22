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
def makeFeatureVariations(self, table, table_tag):
    feature_vars = {}
    has_any_variations = False
    for (_, _, feature_tag), variations in self.feature_variations_.items():
        feature_vars[feature_tag] = []
        for conditionset, builders in variations.items():
            raw_conditionset = self.conditionsets_[conditionset]
            indices = []
            for b in builders:
                if b.table != table_tag:
                    continue
                assert b.lookup_index is not None
                indices.append(b.lookup_index)
                has_any_variations = True
            feature_vars[feature_tag].append((raw_conditionset, indices))
    if has_any_variations:
        for feature_tag, conditions_and_lookups in feature_vars.items():
            addFeatureVariationsRaw(self.font, table, conditions_and_lookups, feature_tag)