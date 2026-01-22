from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def buildSinglePos(mapping, glyphMap):
    """Builds a list of single adjustment (GPOS1) subtables.

    This builds a list of SinglePos subtables from a dictionary of glyph
    names and their positioning adjustments. The format of the subtables are
    determined to optimize the size of the resulting subtables.
    See also :func:`buildSinglePosSubtable`.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.SinglePosBuilder` instead.

    Example::

        mapping = {
            "V": buildValue({ "xAdvance" : +5 }),
            # ...
        }

        subtables = buildSinglePos(pairs, font.getReverseGlyphMap())

    Args:
        mapping (dict): A mapping between glyphnames and
            ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A list of ``otTables.SinglePos`` objects.
    """
    result, handled = ([], set())
    coverages, masks, values = ({}, {}, {})
    for glyph, value in mapping.items():
        key = _getSinglePosValueKey(value)
        coverages.setdefault(key, []).append(glyph)
        masks.setdefault(key[0], []).append(key)
        values[key] = value
    for key, glyphs in coverages.items():
        if len(glyphs) * _getSinglePosValueSize(key) > 5:
            format1Mapping = {g: values[key] for g in glyphs}
            result.append(buildSinglePosSubtable(format1Mapping, glyphMap))
            handled.add(key)
    for valueFormat, keys in masks.items():
        f2 = [k for k in keys if k not in handled]
        if len(f2) > 1:
            format2Mapping = {}
            for k in f2:
                format2Mapping.update(((g, values[k]) for g in coverages[k]))
            result.append(buildSinglePosSubtable(format2Mapping, glyphMap))
            handled.update(f2)
    for key, glyphs in coverages.items():
        if key not in handled:
            for g in glyphs:
                st = buildSinglePosSubtable({g: values[key]}, glyphMap)
            result.append(st)
    result.sort(key=lambda t: _getSinglePosTableKey(t, glyphMap))
    return result