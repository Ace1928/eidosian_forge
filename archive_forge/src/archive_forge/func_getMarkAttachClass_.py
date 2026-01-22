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
def getMarkAttachClass_(self, location, glyphs):
    glyphs = frozenset(glyphs)
    id_ = self.markAttachClassID_.get(glyphs)
    if id_ is not None:
        return id_
    id_ = len(self.markAttachClassID_) + 1
    self.markAttachClassID_[glyphs] = id_
    for glyph in glyphs:
        if glyph in self.markAttach_:
            _, loc = self.markAttach_[glyph]
            raise FeatureLibError('Glyph %s already has been assigned a MarkAttachmentType at %s' % (glyph, loc), location)
        self.markAttach_[glyph] = (id_, location)
    return id_