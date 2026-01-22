from fontTools.misc import sstruct
from fontTools.misc.textTools import readHex, safeEval
import struct
def is_reference_type(self):
    """Returns True if this glyph is a reference to another glyph's image data."""
    return self.graphicType == 'dupe' or self.graphicType == 'flip'