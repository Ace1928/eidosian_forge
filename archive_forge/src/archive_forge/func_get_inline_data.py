import logging
import re
from io import BytesIO
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .pdfcolor import PDFColorSpace
from .pdfcolor import PREDEFINED_COLORSPACE
from .pdfdevice import PDFDevice
from .pdfdevice import PDFTextSeq
from .pdffont import PDFCIDFont
from .pdffont import PDFFont
from .pdffont import PDFFontError
from .pdffont import PDFTrueTypeFont
from .pdffont import PDFType1Font
from .pdffont import PDFType3Font
from .pdfpage import PDFPage
from .pdftypes import PDFException
from .pdftypes import PDFObjRef
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import list_value
from .pdftypes import resolve1
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral, PSTypeError
from .psparser import PSStackParser
from .psparser import PSStackType
from .psparser import keyword_name
from .psparser import literal_name
from .utils import MATRIX_IDENTITY
from .utils import Matrix, Point, PathSegment, Rect
from .utils import choplist
from .utils import mult_matrix
def get_inline_data(self, pos: int, target: bytes=b'EI') -> Tuple[int, bytes]:
    self.seek(pos)
    i = 0
    data = b''
    while i <= len(target):
        self.fillbuf()
        if i:
            ci = self.buf[self.charpos]
            c = bytes((ci,))
            data += c
            self.charpos += 1
            if len(target) <= i and c.isspace():
                i += 1
            elif i < len(target) and c == bytes((target[i],)):
                i += 1
            else:
                i = 0
        else:
            try:
                j = self.buf.index(target[0], self.charpos)
                data += self.buf[self.charpos:j + 1]
                self.charpos = j + 1
                i = 1
            except ValueError:
                data += self.buf[self.charpos:]
                self.charpos = len(self.buf)
    data = data[:-(len(target) + 1)]
    data = re.sub(b'(\\x0d\\x0a|[\\x0d\\x0a])$', b'', data)
    return (pos, data)