import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def scan_record(self):
    assert self._opened_for_scan, 'file not open for scanning'
    header = self._fp.read(LEVELDBLOG_HEADER_LEN)
    if len(header) == 0:
        return None
    assert len(header) == LEVELDBLOG_HEADER_LEN, 'record header is {} bytes instead of the expected {}'.format(len(header), LEVELDBLOG_HEADER_LEN)
    fields = struct.unpack('<IHB', header)
    checksum, dlength, dtype = fields
    self._index += LEVELDBLOG_HEADER_LEN
    data = self._fp.read(dlength)
    checksum_computed = zlib.crc32(data, self._crc[dtype]) & 4294967295
    assert checksum == checksum_computed, 'record checksum is invalid, data may be corrupt'
    self._index += dlength
    return (dtype, data)