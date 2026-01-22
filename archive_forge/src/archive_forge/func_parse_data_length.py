import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def parse_data_length(self, segment: JBIG2Segment, length: int, field: bytes) -> int:
    if length:
        if cast(JBIG2SegmentFlags, segment['flags'])['type'] == SEG_TYPE_IMMEDIATE_GEN_REGION and length == DATA_LEN_UNKNOWN:
            raise NotImplementedError('Working with unknown segment length is not implemented yet')
        else:
            segment['raw_data'] = self.stream.read(length)
    return length