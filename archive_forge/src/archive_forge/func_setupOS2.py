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
def setupOS2(self, **values):
    """Create a new `OS/2` table and initialize it with default values,
        which can be overridden by keyword arguments.
        """
    self._initTableWithValues('OS/2', _OS2Defaults, values)
    if 'xAvgCharWidth' not in values:
        assert 'hmtx' in self.font, "the 'hmtx' table must be setup before the 'OS/2' table"
        self.font['OS/2'].recalcAvgCharWidth(self.font)
    if not ('ulUnicodeRange1' in values or 'ulUnicodeRange2' in values or 'ulUnicodeRange3' in values or ('ulUnicodeRange3' in values)):
        assert 'cmap' in self.font, "the 'cmap' table must be setup before the 'OS/2' table"
        self.font['OS/2'].recalcUnicodeRanges(self.font)