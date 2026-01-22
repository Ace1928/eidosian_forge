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
def build_codepages_(self, pages):
    pages2bits = {1252: 0, 1250: 1, 1251: 2, 1253: 3, 1254: 4, 1255: 5, 1256: 6, 1257: 7, 1258: 8, 874: 16, 932: 17, 936: 18, 949: 19, 950: 20, 1361: 21, 869: 48, 866: 49, 865: 50, 864: 51, 863: 52, 862: 53, 861: 54, 860: 55, 857: 56, 855: 57, 852: 58, 775: 59, 737: 60, 708: 61, 850: 62, 437: 63}
    bits = [pages2bits[p] for p in pages if p in pages2bits]
    pages = []
    for i in range(2):
        pages.append('')
        for j in range(i * 32, (i + 1) * 32):
            if j in bits:
                pages[i] += '1'
            else:
                pages[i] += '0'
    return [binary2num(p[::-1]) for p in pages]