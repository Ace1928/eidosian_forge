import itertools
import logging
import re
import struct
from hashlib import sha256, md5, sha384, sha512
from typing import (
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from . import settings
from .arcfour import Arcfour
from .data_structures import NumberTree
from .pdfparser import PDFSyntaxError, PDFParser, PDFStreamParser
from .pdftypes import (
from .psparser import PSEOF, literal_name, LIT, KWD
from .utils import choplist, decode_text, nunpack, format_int_roman, format_int_alpha
def load_trailer(self, parser: PDFParser) -> None:
    try:
        _, kwd = parser.nexttoken()
        assert kwd is KWD(b'trailer'), str(kwd)
        _, dic = parser.nextobject()
    except PSEOF:
        x = parser.pop(1)
        if not x:
            raise PDFNoValidXRef('Unexpected EOF - file corrupted')
        _, dic = x[0]
    self.trailer.update(dict_value(dic))
    log.debug('trailer=%r', self.trailer)