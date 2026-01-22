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
def setupPost(self, keepGlyphNames=True, **values):
    """Create a new `post` table and initialize it with default values,
        which can be overridden by keyword arguments.
        """
    isCFF2 = 'CFF2' in self.font
    postTable = self._initTableWithValues('post', _postDefaults, values)
    if (self.isTTF or isCFF2) and keepGlyphNames:
        postTable.formatType = 2.0
        postTable.extraNames = []
        postTable.mapping = {}
    else:
        postTable.formatType = 3.0