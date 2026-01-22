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
def build_vhea(self):
    if not self.vhea_:
        return
    table = self.font.get('vhea')
    if not table:
        table = self.font['vhea'] = newTable('vhea')
        table.decompile(b'\x00' * 36, self.font)
        table.tableVersion = 69632
    if 'verttypoascender' in self.vhea_:
        table.ascent = self.vhea_['verttypoascender']
    if 'verttypodescender' in self.vhea_:
        table.descent = self.vhea_['verttypodescender']
    if 'verttypolinegap' in self.vhea_:
        table.lineGap = self.vhea_['verttypolinegap']