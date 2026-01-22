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
def compute_u(self, key: bytes) -> bytes:
    if self.r == 2:
        return Arcfour(key).encrypt(self.PASSWORD_PADDING)
    else:
        hash = md5(self.PASSWORD_PADDING)
        hash.update(self.docid[0])
        result = Arcfour(key).encrypt(hash.digest())
        for i in range(1, 20):
            k = b''.join((bytes((c ^ i,)) for c in iter(key)))
            result = Arcfour(k).encrypt(result)
        result += result
        return result