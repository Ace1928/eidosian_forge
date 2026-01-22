from typing import List
from fontTools.misc.vector import Vector
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.fixedTools import floatToFixed as fl2fi
from fontTools.misc.textTools import Tag, tostr
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis, NamedInstance
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates, dropImpliedOnCurvePoints
from fontTools.ttLib.tables.ttProgram import Program
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.merger import VariationMerger, COLRVariationMerger
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta_optimize
from fontTools.varLib.featureVars import addFeatureVariations
from fontTools.designspaceLib import DesignSpaceDocument, InstanceDescriptor
from fontTools.designspaceLib.split import splitInterpolable, splitVariableFonts
from fontTools.varLib.stat import buildVFStatTable
from fontTools.colorLib.builder import buildColrV1
from fontTools.colorLib.unbuilder import unbuildColrV1
from functools import partial
from collections import OrderedDict, defaultdict, namedtuple
import os.path
import logging
from copy import deepcopy
from pprint import pformat
from re import fullmatch
from .errors import VarLibError, VarLibValidationError
def build_many(designspace: DesignSpaceDocument, master_finder=lambda s: s, exclude=[], optimize=True, skip_vf=lambda vf_name: False, colr_layer_reuse=True, drop_implied_oncurves=False):
    """
    Build variable fonts from a designspace file, version 5 which can define
    several VFs, or version 4 which has implicitly one VF covering the whole doc.

    If master_finder is set, it should be a callable that takes master
    filename as found in designspace file and map it to master font
    binary as to be opened (eg. .ttf or .otf).

    skip_vf can be used to skip building some of the variable fonts defined in
    the input designspace. It's a predicate that takes as argument the name
    of the variable font and returns `bool`.

    Always returns a Dict[str, TTFont] keyed by VariableFontDescriptor.name
    """
    res = {}
    doBuildStatFromDSv5 = 'STAT' not in exclude and designspace.formatTuple >= (5, 0) and (any((a.axisLabels or a.axisOrdering is not None for a in designspace.axes)) or designspace.locationLabels)
    for _location, subDoc in splitInterpolable(designspace):
        for name, vfDoc in splitVariableFonts(subDoc):
            if skip_vf(name):
                log.debug(f'Skipping variable TTF font: {name}')
                continue
            vf = build(vfDoc, master_finder, exclude=exclude, optimize=optimize, colr_layer_reuse=colr_layer_reuse, drop_implied_oncurves=drop_implied_oncurves)[0]
            if doBuildStatFromDSv5:
                buildVFStatTable(vf, designspace, name)
            res[name] = vf
    return res