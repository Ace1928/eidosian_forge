import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
def validate_crc(self):
    assert self._decompressed is False, 'Validate should be called before iteration'
    crc = self.crc
    data_view = memoryview(self._buffer)[self.ATTRIBUTES_OFFSET:]
    verify_crc = calc_crc32c(data_view.tobytes())
    return crc == verify_crc