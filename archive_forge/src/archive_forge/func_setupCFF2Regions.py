from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from .ttLib import TTFont, newTable
from .ttLib.tables._c_m_a_p import cmap_classes
from .ttLib.tables._g_l_y_f import flagCubic
from .ttLib.tables.O_S_2f_2 import Panose
from .misc.timeTools import timestampNow
import struct
from collections import OrderedDict
def setupCFF2Regions(self, regions):
    from .varLib.builder import buildVarRegionList, buildVarData, buildVarStore
    from .cffLib import VarStoreData
    assert 'fvar' in self.font, 'fvar must to be set up first'
    assert 'CFF2' in self.font, 'CFF2 must to be set up first'
    axisTags = [a.axisTag for a in self.font['fvar'].axes]
    varRegionList = buildVarRegionList(regions, axisTags)
    varData = buildVarData(list(range(len(regions))), None, optimize=False)
    varStore = buildVarStore(varRegionList, [varData])
    vstore = VarStoreData(otVarStore=varStore)
    topDict = self.font['CFF2'].cff.topDictIndex[0]
    topDict.VarStore = vstore
    for fontDict in topDict.FDArray:
        fontDict.Private.vstore = vstore