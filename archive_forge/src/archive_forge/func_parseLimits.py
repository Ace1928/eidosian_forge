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
def parseLimits(limits: Iterable[str]) -> Dict[str, Optional[AxisTriple]]:
    result = {}
    for limitString in limits:
        match = re.match('^(\\w{1,4})=(?:(drop)|(?:([^:]*)(?:[:]([^:]*))?(?:[:]([^:]*))?))$', limitString)
        if not match:
            raise ValueError('invalid location format: %r' % limitString)
        tag = match.group(1).ljust(4)
        if match.group(2):
            result[tag] = None
            continue
        triple = match.group(3, 4, 5)
        if triple[1] is None:
            triple = (triple[0], triple[0], triple[0])
        elif triple[2] is None:
            triple = (triple[0], None, triple[1])
        triple = tuple((float(v) if v else None for v in triple))
        result[tag] = AxisTriple(*triple)
    return result