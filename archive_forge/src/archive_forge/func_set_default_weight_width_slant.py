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
def set_default_weight_width_slant(font, location):
    if 'OS/2' in font:
        if 'wght' in location:
            weight_class = otRound(max(1, min(location['wght'], 1000)))
            if font['OS/2'].usWeightClass != weight_class:
                log.info('Setting OS/2.usWeightClass = %s', weight_class)
                font['OS/2'].usWeightClass = weight_class
        if 'wdth' in location:
            widthValue = min(max(location['wdth'], 50), 200)
            widthClass = otRound(models.piecewiseLinearMap(widthValue, WDTH_VALUE_TO_OS2_WIDTH_CLASS))
            if font['OS/2'].usWidthClass != widthClass:
                log.info('Setting OS/2.usWidthClass = %s', widthClass)
                font['OS/2'].usWidthClass = widthClass
    if 'slnt' in location and 'post' in font:
        italicAngle = max(-90, min(location['slnt'], 90))
        if font['post'].italicAngle != italicAngle:
            log.info('Setting post.italicAngle = %s', italicAngle)
            font['post'].italicAngle = italicAngle