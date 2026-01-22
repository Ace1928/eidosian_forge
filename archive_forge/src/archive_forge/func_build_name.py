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
def build_name(self):
    if not self.names_:
        return
    table = self.font.get('name')
    if not table:
        table = self.font['name'] = newTable('name')
        table.names = []
    for name in self.names_:
        nameID, platformID, platEncID, langID, string = name
        if not isinstance(nameID, int):
            tag = nameID
            if tag in self.featureNames_:
                if tag not in self.featureNames_ids_:
                    self.featureNames_ids_[tag] = self.get_user_name_id(table)
                    assert self.featureNames_ids_[tag] is not None
                nameID = self.featureNames_ids_[tag]
            elif tag[0] in self.cv_parameters_:
                if tag not in self.cv_parameters_ids_:
                    self.cv_parameters_ids_[tag] = self.get_user_name_id(table)
                    assert self.cv_parameters_ids_[tag] is not None
                nameID = self.cv_parameters_ids_[tag]
        table.setName(string, nameID, platformID, platEncID, langID)
    table.names.sort()