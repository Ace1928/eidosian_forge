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
def setupCFF(self, psName, fontInfo, charStringsDict, privateDict):
    from .cffLib import CFFFontSet, TopDictIndex, TopDict, CharStrings, GlobalSubrsIndex, PrivateDict
    assert not self.isTTF
    self.font.sfntVersion = 'OTTO'
    fontSet = CFFFontSet()
    fontSet.major = 1
    fontSet.minor = 0
    fontSet.otFont = self.font
    fontSet.fontNames = [psName]
    fontSet.topDictIndex = TopDictIndex()
    globalSubrs = GlobalSubrsIndex()
    fontSet.GlobalSubrs = globalSubrs
    private = PrivateDict()
    for key, value in privateDict.items():
        setattr(private, key, value)
    fdSelect = None
    fdArray = None
    topDict = TopDict()
    topDict.charset = self.font.getGlyphOrder()
    topDict.Private = private
    topDict.GlobalSubrs = fontSet.GlobalSubrs
    for key, value in fontInfo.items():
        setattr(topDict, key, value)
    if 'FontMatrix' not in fontInfo:
        scale = 1 / self.font['head'].unitsPerEm
        topDict.FontMatrix = [scale, 0, 0, scale, 0, 0]
    charStrings = CharStrings(None, topDict.charset, globalSubrs, private, fdSelect, fdArray)
    for glyphName, charString in charStringsDict.items():
        charString.private = private
        charString.globalSubrs = globalSubrs
        charStrings[glyphName] = charString
    topDict.CharStrings = charStrings
    fontSet.topDictIndex.append(topDict)
    self.font['CFF '] = newTable('CFF ')
    self.font['CFF '].cff = fontSet