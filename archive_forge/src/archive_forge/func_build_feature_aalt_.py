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
def build_feature_aalt_(self):
    if not self.aalt_features_ and (not self.aalt_alternates_):
        return
    alternates = {g: list(a) for g, a in self.aalt_alternates_.items()}
    for location, name in self.aalt_features_ + [(None, 'aalt')]:
        feature = [(script, lang, feature, lookups) for (script, lang, feature), lookups in self.features_.items() if feature == name]
        if not feature and name != 'aalt':
            warnings.warn('%s: Feature %s has not been defined' % (location, name))
            continue
        for script, lang, feature, lookups in feature:
            for lookuplist in lookups:
                if not isinstance(lookuplist, list):
                    lookuplist = [lookuplist]
                for lookup in lookuplist:
                    for glyph, alts in lookup.getAlternateGlyphs().items():
                        alts_for_glyph = alternates.setdefault(glyph, [])
                        alts_for_glyph.extend((g for g in alts if g not in alts_for_glyph))
    single = {glyph: repl[0] for glyph, repl in alternates.items() if len(repl) == 1}
    multi = {glyph: repl for glyph, repl in alternates.items() if len(repl) > 1}
    if not single and (not multi):
        return
    self.features_ = {(script, lang, feature): lookups for (script, lang, feature), lookups in self.features_.items() if feature != 'aalt'}
    old_lookups = self.lookups_
    self.lookups_ = []
    self.start_feature(self.aalt_location_, 'aalt')
    if single:
        single_lookup = self.get_lookup_(location, SingleSubstBuilder)
        single_lookup.mapping = single
    if multi:
        multi_lookup = self.get_lookup_(location, AlternateSubstBuilder)
        multi_lookup.alternates = multi
    self.end_feature()
    self.lookups_.extend(old_lookups)