import codecs
from typing import Dict, List, Tuple, Union
from .._codecs import _pdfdoc_encoding
from .._utils import StreamType, b_, logger_warning, read_non_whitespace
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfStreamError
from ._base import ByteStringObject, TextStringObject
def decode_pdfdocencoding(byte_array: bytes) -> str:
    retval = ''
    for b in byte_array:
        c = _pdfdoc_encoding[b]
        if c == '\x00':
            raise UnicodeDecodeError('pdfdocencoding', bytearray(b), -1, -1, 'does not exist in translation table')
        retval += c
    return retval