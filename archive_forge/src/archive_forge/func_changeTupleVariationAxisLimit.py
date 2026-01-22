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
def changeTupleVariationAxisLimit(var, axisTag, axisLimit):
    assert isinstance(axisLimit, NormalizedAxisTripleAndDistances)
    lower, peak, upper = var.axes.get(axisTag, (-1, 0, 1))
    if peak == 0:
        if axisTag in var.axes:
            del var.axes[axisTag]
        return [var]
    if not lower <= peak <= upper or (lower < 0 and upper > 0):
        return []
    if axisTag not in var.axes:
        return [var]
    tent = var.axes[axisTag]
    solutions = solver.rebaseTent(tent, axisLimit)
    out = []
    for scalar, tent in solutions:
        newVar = TupleVariation(var.axes, var.coordinates) if len(solutions) > 1 else var
        if tent is None:
            newVar.axes.pop(axisTag)
        else:
            assert tent[1] != 0, tent
            newVar.axes[axisTag] = tent
        newVar *= scalar
        out.append(newVar)
    return out