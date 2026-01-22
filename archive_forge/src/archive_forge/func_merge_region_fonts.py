from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
def merge_region_fonts(varFont, model, ordered_fonts_list, glyphOrder):
    topDict = varFont['CFF2'].cff.topDictIndex[0]
    top_dicts = [topDict] + [_cff_or_cff2(ttFont).cff.topDictIndex[0] for ttFont in ordered_fonts_list[1:]]
    num_masters = len(model.mapping)
    cvData = merge_charstrings(glyphOrder, num_masters, top_dicts, model)
    fd_map = getfd_map(varFont, ordered_fonts_list)
    merge_PrivateDicts(top_dicts, cvData.vsindex_dict, model, fd_map)
    addCFFVarStore(varFont, model, cvData.varDataList, cvData.masterSupports)