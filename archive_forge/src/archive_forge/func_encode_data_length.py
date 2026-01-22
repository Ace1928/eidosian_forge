import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def encode_data_length(self, value: int, segment: JBIG2Segment) -> bytes:
    data = pack('>L', value)
    data += cast(bytes, segment['raw_data'])
    return data