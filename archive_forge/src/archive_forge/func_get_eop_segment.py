import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def get_eop_segment(self, seg_number: int, page_number: int) -> JBIG2Segment:
    return {'data_length': 0, 'flags': {'deferred': False, 'type': SEG_TYPE_END_OF_PAGE}, 'number': seg_number, 'page_assoc': page_number, 'raw_data': b'', 'retention_flags': JBIG2StreamWriter.EMPTY_RETENTION_FLAGS}