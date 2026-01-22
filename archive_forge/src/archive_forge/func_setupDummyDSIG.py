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
def setupDummyDSIG(self):
    """This adds an empty DSIG table to the font to make some MS applications
        happy. This does not properly sign the font.
        """
    values = dict(ulVersion=1, usFlag=0, usNumSigs=0, signatureRecords=[])
    self._initTableWithValues('DSIG', {}, values)