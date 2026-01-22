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
def getobj(self, objid: int) -> object:
    """Get object from PDF

        :raises PDFException if PDFDocument is not initialized
        :raises PDFObjectNotFound if objid does not exist in PDF
        """
    if not self.xrefs:
        raise PDFException('PDFDocument is not initialized')
    log.debug('getobj: objid=%r', objid)
    if objid in self._cached_objs:
        obj, genno = self._cached_objs[objid]
    else:
        for xref in self.xrefs:
            try:
                strmid, index, genno = xref.get_pos(objid)
            except KeyError:
                continue
            try:
                if strmid is not None:
                    stream = stream_value(self.getobj(strmid))
                    obj = self._getobj_objstm(stream, index, objid)
                else:
                    obj = self._getobj_parse(index, objid)
                    if self.decipher:
                        obj = decipher_all(self.decipher, objid, genno, obj)
                if isinstance(obj, PDFStream):
                    obj.set_objid(objid, genno)
                break
            except (PSEOF, PDFSyntaxError):
                continue
        else:
            raise PDFObjectNotFound(objid)
        log.debug('register: objid=%r: %r', objid, obj)
        if self.caching:
            self._cached_objs[objid] = (obj, genno)
    return obj