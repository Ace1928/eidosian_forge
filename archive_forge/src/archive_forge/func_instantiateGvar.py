from fontTools.misc.fixedTools import (
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import _g_l_y_f
from fontTools import varLib
from fontTools import subset  # noqa: F401
from fontTools.varLib import builder
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.instancer import names
from .featureVars import instantiateFeatureVariations
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.varLib.instancer import solver
import collections
import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import logging
import os
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings
def instantiateGvar(varfont, axisLimits, optimize=True):
    log.info('Instantiating glyf/gvar tables')
    gvar = varfont['gvar']
    glyf = varfont['glyf']
    hMetrics = varfont['hmtx'].metrics
    vMetrics = getattr(varfont.get('vmtx'), 'metrics', None)
    glyphnames = sorted(glyf.glyphOrder, key=lambda name: (glyf[name].getCompositeMaxpValues(glyf).maxComponentDepth if glyf[name].isComposite() or glyf[name].isVarComposite() else 0, name))
    for glyphname in glyphnames:
        _instantiateGvarGlyph(glyphname, glyf, gvar, hMetrics, vMetrics, axisLimits, optimize=optimize)
    if not gvar.variations:
        del varfont['gvar']