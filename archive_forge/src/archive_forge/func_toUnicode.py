from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import newTable
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools import ttLib
import fontTools.ttLib.tables.otTables as otTables
from fontTools.ttLib.tables import C_P_A_L_
from . import DefaultTable
import struct
import logging
def toUnicode(self, errors='strict'):
    """
        If self.string is a Unicode string, return it; otherwise try decoding the
        bytes in self.string to a Unicode string using the encoding of this
        entry as returned by self.getEncoding(); Note that  self.getEncoding()
        returns 'ascii' if the encoding is unknown to the library.

        Certain heuristics are performed to recover data from bytes that are
        ill-formed in the chosen encoding, or that otherwise look misencoded
        (mostly around bad UTF-16BE encoded bytes, or bytes that look like UTF-16BE
        but marked otherwise).  If the bytes are ill-formed and the heuristics fail,
        the error is handled according to the errors parameter to this function, which is
        passed to the underlying decode() function; by default it throws a
        UnicodeDecodeError exception.

        Note: The mentioned heuristics mean that roundtripping a font to XML and back
        to binary might recover some misencoded data whereas just loading the font
        and saving it back will not change them.
        """

    def isascii(b):
        return b >= 32 and b <= 126 or b in [9, 10, 13]
    encoding = self.getEncoding()
    string = self.string
    if isinstance(string, bytes) and encoding == 'utf_16_be' and (len(string) % 2 == 1):
        if byteord(string[-1]) == 0:
            string = string[:-1]
        elif all((byteord(b) == 0 if i % 2 else isascii(byteord(b)) for i, b in enumerate(string))):
            string = b'\x00' + string
        elif byteord(string[0]) == 0 and all((isascii(byteord(b)) for b in string[1:])):
            string = bytesjoin((b'\x00' + bytechr(byteord(b)) for b in string[1:]))
    string = tostr(string, encoding=encoding, errors=errors)
    if all((ord(c) == 0 if i % 2 == 0 else isascii(ord(c)) for i, c in enumerate(string))):
        string = ''.join((c for c in string[1::2]))
    return string