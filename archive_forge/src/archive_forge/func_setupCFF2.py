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
def setupCFF2(self, charStringsDict, fdArrayList=None, regions=None):
    from .cffLib import CFFFontSet, TopDictIndex, TopDict, CharStrings, GlobalSubrsIndex, PrivateDict, FDArrayIndex, FontDict
    assert not self.isTTF
    self.font.sfntVersion = 'OTTO'
    fontSet = CFFFontSet()
    fontSet.major = 2
    fontSet.minor = 0
    cff2GetGlyphOrder = self.font.getGlyphOrder
    fontSet.topDictIndex = TopDictIndex(None, cff2GetGlyphOrder, None)
    globalSubrs = GlobalSubrsIndex()
    fontSet.GlobalSubrs = globalSubrs
    if fdArrayList is None:
        fdArrayList = [{}]
    fdSelect = None
    fdArray = FDArrayIndex()
    fdArray.strings = None
    fdArray.GlobalSubrs = globalSubrs
    for privateDict in fdArrayList:
        fontDict = FontDict()
        fontDict.setCFF2(True)
        private = PrivateDict()
        for key, value in privateDict.items():
            setattr(private, key, value)
        fontDict.Private = private
        fdArray.append(fontDict)
    topDict = TopDict()
    topDict.cff2GetGlyphOrder = cff2GetGlyphOrder
    topDict.FDArray = fdArray
    scale = 1 / self.font['head'].unitsPerEm
    topDict.FontMatrix = [scale, 0, 0, scale, 0, 0]
    private = fdArray[0].Private
    charStrings = CharStrings(None, None, globalSubrs, private, fdSelect, fdArray)
    for glyphName, charString in charStringsDict.items():
        charString.private = private
        charString.globalSubrs = globalSubrs
        charStrings[glyphName] = charString
    topDict.CharStrings = charStrings
    fontSet.topDictIndex.append(topDict)
    self.font['CFF2'] = newTable('CFF2')
    self.font['CFF2'].cff = fontSet
    if regions:
        self.setupCFF2Regions(regions)