import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def parse_retention_flags(self, segment: JBIG2Segment, flags: int, field: bytes) -> JBIG2RetentionFlags:
    ref_count = masked_value(REF_COUNT_SHORT_MASK, flags)
    retain_segments = []
    ref_segments = []
    if ref_count < REF_COUNT_LONG:
        for bit_pos in range(5):
            retain_segments.append(bit_set(bit_pos, flags))
    else:
        field += self.stream.read(3)
        ref_count = unpack_int('>L', field)
        ref_count = masked_value(REF_COUNT_LONG_MASK, ref_count)
        ret_bytes_count = int(math.ceil((ref_count + 1) / 8))
        for ret_byte_index in range(ret_bytes_count):
            ret_byte = unpack_int('>B', self.stream.read(1))
            for bit_pos in range(7):
                retain_segments.append(bit_set(bit_pos, ret_byte))
    seg_num = segment['number']
    assert isinstance(seg_num, int)
    if seg_num <= 256:
        ref_format = '>B'
    elif seg_num <= 65536:
        ref_format = '>I'
    else:
        ref_format = '>L'
    ref_size = calcsize(ref_format)
    for ref_index in range(ref_count):
        ref_data = self.stream.read(ref_size)
        ref = unpack_int(ref_format, ref_data)
        ref_segments.append(ref)
    return {'ref_count': ref_count, 'retain_segments': retain_segments, 'ref_segments': ref_segments}