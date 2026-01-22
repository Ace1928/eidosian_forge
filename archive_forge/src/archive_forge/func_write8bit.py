from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def write8bit(self, data, strip=False):
    """Writes a bytes() sequence into the XML, escaping
        non-ASCII bytes.  When this is read in xmlReader,
        the original bytes can be recovered by encoding to
        'latin-1'."""
    self._writeraw(escape8bit(data), strip=strip)