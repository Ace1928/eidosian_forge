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
def build_OS_2(self):
    if not self.os2_:
        return
    table = self.font.get('OS/2')
    if not table:
        table = self.font['OS/2'] = newTable('OS/2')
        data = b'\x00' * sstruct.calcsize(getTableModule('OS/2').OS2_format_0)
        table.decompile(data, self.font)
    version = 0
    if 'fstype' in self.os2_:
        table.fsType = self.os2_['fstype']
    if 'panose' in self.os2_:
        panose = getTableModule('OS/2').Panose()
        panose.bFamilyType, panose.bSerifStyle, panose.bWeight, panose.bProportion, panose.bContrast, panose.bStrokeVariation, panose.bArmStyle, panose.bLetterForm, panose.bMidline, panose.bXHeight = self.os2_['panose']
        table.panose = panose
    if 'typoascender' in self.os2_:
        table.sTypoAscender = self.os2_['typoascender']
    if 'typodescender' in self.os2_:
        table.sTypoDescender = self.os2_['typodescender']
    if 'typolinegap' in self.os2_:
        table.sTypoLineGap = self.os2_['typolinegap']
    if 'winascent' in self.os2_:
        table.usWinAscent = self.os2_['winascent']
    if 'windescent' in self.os2_:
        table.usWinDescent = self.os2_['windescent']
    if 'vendor' in self.os2_:
        table.achVendID = safeEval("'''" + self.os2_['vendor'] + "'''")
    if 'weightclass' in self.os2_:
        table.usWeightClass = self.os2_['weightclass']
    if 'widthclass' in self.os2_:
        table.usWidthClass = self.os2_['widthclass']
    if 'unicoderange' in self.os2_:
        table.setUnicodeRanges(self.os2_['unicoderange'])
    if 'codepagerange' in self.os2_:
        pages = self.build_codepages_(self.os2_['codepagerange'])
        table.ulCodePageRange1, table.ulCodePageRange2 = pages
        version = 1
    if 'xheight' in self.os2_:
        table.sxHeight = self.os2_['xheight']
        version = 2
    if 'capheight' in self.os2_:
        table.sCapHeight = self.os2_['capheight']
        version = 2
    if 'loweropsize' in self.os2_:
        table.usLowerOpticalPointSize = self.os2_['loweropsize']
        version = 5
    if 'upperopsize' in self.os2_:
        table.usUpperOpticalPointSize = self.os2_['upperopsize']
        version = 5

    def checkattr(table, attrs):
        for attr in attrs:
            if not hasattr(table, attr):
                setattr(table, attr, 0)
    table.version = max(version, table.version)
    if version >= 1:
        checkattr(table, ('ulCodePageRange1', 'ulCodePageRange2'))
    if version >= 2:
        checkattr(table, ('sxHeight', 'sCapHeight', 'usDefaultChar', 'usBreakChar', 'usMaxContext'))
    if version >= 5:
        checkattr(table, ('usLowerOpticalPointSize', 'usUpperOpticalPointSize'))