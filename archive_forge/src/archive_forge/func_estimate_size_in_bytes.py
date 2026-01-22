import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
@classmethod
def estimate_size_in_bytes(cls, key, value, headers):
    """ Get the upper bound estimate on the size of record
        """
    return cls.HEADER_STRUCT.size + cls.MAX_RECORD_OVERHEAD + cls.size_of(key, value, headers)