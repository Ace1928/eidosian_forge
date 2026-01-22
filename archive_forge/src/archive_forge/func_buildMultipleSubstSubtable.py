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
def buildMultipleSubstSubtable(mapping):
    """Builds a multiple substitution (GSUB2) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.MultipleSubstBuilder` instead.

    Example::

        # sub uni06C0 by uni06D5.fina hamza.above
        # sub uni06C2 by uni06C1.fina hamza.above;

        subtable = buildMultipleSubstSubtable({
            "uni06C0": [ "uni06D5.fina", "hamza.above"],
            "uni06C2": [ "uni06D1.fina", "hamza.above"]
        })

    Args:
        mapping: A dictionary mapping input glyph names to a list of output
            glyph names.

    Returns:
        An ``otTables.MultipleSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.MultipleSubst()
    self.mapping = dict(mapping)
    return self