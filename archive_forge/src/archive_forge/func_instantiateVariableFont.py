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
def instantiateVariableFont(varfont, axisLimits, inplace=False, optimize=True, overlap=OverlapMode.KEEP_AND_SET_FLAGS, updateFontNames=False):
    """Instantiate variable font, either fully or partially.

    Depending on whether the `axisLimits` dictionary references all or some of the
    input varfont's axes, the output font will either be a full instance (static
    font) or a variable font with possibly less variation data.

    Args:
        varfont: a TTFont instance, which must contain at least an 'fvar' table.
            Note that variable fonts with 'CFF2' table are not supported yet.
        axisLimits: a dict keyed by axis tags (str) containing the coordinates (float)
            along one or more axes where the desired instance will be located.
            If the value is `None`, the default coordinate as per 'fvar' table for
            that axis is used.
            The limit values can also be (min, max) tuples for restricting an
            axis's variation range. The default axis value must be included in
            the new range.
        inplace (bool): whether to modify input TTFont object in-place instead of
            returning a distinct object.
        optimize (bool): if False, do not perform IUP-delta optimization on the
            remaining 'gvar' table's deltas. Possibly faster, and might work around
            rendering issues in some buggy environments, at the cost of a slightly
            larger file size.
        overlap (OverlapMode): variable fonts usually contain overlapping contours, and
            some font rendering engines on Apple platforms require that the
            `OVERLAP_SIMPLE` and `OVERLAP_COMPOUND` flags in the 'glyf' table be set to
            force rendering using a non-zero fill rule. Thus we always set these flags
            on all glyphs to maximise cross-compatibility of the generated instance.
            You can disable this by passing OverlapMode.KEEP_AND_DONT_SET_FLAGS.
            If you want to remove the overlaps altogether and merge overlapping
            contours and components, you can pass OverlapMode.REMOVE (or
            REMOVE_AND_IGNORE_ERRORS to not hard-fail on tricky glyphs). Note that this
            requires the skia-pathops package (available to pip install).
            The overlap parameter only has effect when generating full static instances.
        updateFontNames (bool): if True, update the instantiated font's name table using
            the Axis Value Tables from the STAT table. The name table and the style bits
            in the head and OS/2 table will be updated so they conform to the R/I/B/BI
            model. If the STAT table is missing or an Axis Value table is missing for
            a given axis coordinate, a ValueError will be raised.
    """
    overlap = OverlapMode(int(overlap))
    sanityCheckVariableTables(varfont)
    axisLimits = AxisLimits(axisLimits).limitAxesAndPopulateDefaults(varfont)
    log.info('Restricted limits: %s', axisLimits)
    normalizedLimits = axisLimits.normalize(varfont)
    log.info('Normalized limits: %s', normalizedLimits)
    if not inplace:
        varfont = deepcopy(varfont)
    if 'DSIG' in varfont:
        del varfont['DSIG']
    if updateFontNames:
        log.info('Updating name table')
        names.updateNameTable(varfont, axisLimits)
    if 'gvar' in varfont:
        instantiateGvar(varfont, normalizedLimits, optimize=optimize)
    if 'cvar' in varfont:
        instantiateCvar(varfont, normalizedLimits)
    if 'MVAR' in varfont:
        instantiateMVAR(varfont, normalizedLimits)
    if 'HVAR' in varfont:
        instantiateHVAR(varfont, normalizedLimits)
    if 'VVAR' in varfont:
        instantiateVVAR(varfont, normalizedLimits)
    instantiateOTL(varfont, normalizedLimits)
    instantiateFeatureVariations(varfont, normalizedLimits)
    if 'avar' in varfont:
        instantiateAvar(varfont, axisLimits)
    with names.pruningUnusedNames(varfont):
        if 'STAT' in varfont:
            instantiateSTAT(varfont, axisLimits)
        instantiateFvar(varfont, axisLimits)
    if 'fvar' not in varfont:
        if 'glyf' in varfont:
            if overlap == OverlapMode.KEEP_AND_SET_FLAGS:
                setMacOverlapFlags(varfont['glyf'])
            elif overlap in (OverlapMode.REMOVE, OverlapMode.REMOVE_AND_IGNORE_ERRORS):
                from fontTools.ttLib.removeOverlaps import removeOverlaps
                log.info('Removing overlaps from glyf table')
                removeOverlaps(varfont, ignoreErrors=overlap == OverlapMode.REMOVE_AND_IGNORE_ERRORS)
    if 'OS/2' in varfont:
        varfont['OS/2'].recalcAvgCharWidth(varfont)
    varLib.set_default_weight_width_slant(varfont, location=axisLimits.defaultLocation())
    if updateFontNames:
        setRibbiBits(varfont)
    return varfont