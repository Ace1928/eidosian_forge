import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
def extension_processing(self, opcode: Opcode, rsv: RsvBits, payload_len: int) -> None:
    rsv_used = [False, False, False]
    for extension in self.extensions:
        result = extension.frame_inbound_header(self, opcode, rsv, payload_len)
        if isinstance(result, CloseReason):
            raise ParseFailed('error in extension', result)
        for bit, used in enumerate(result):
            if used:
                rsv_used[bit] = True
    for expected, found in zip(rsv_used, rsv):
        if found and (not expected):
            raise ParseFailed('Reserved bit set unexpectedly')