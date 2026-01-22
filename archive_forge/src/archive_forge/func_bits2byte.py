import sys
from io import BytesIO
from typing import Any, List, Tuple, Union, cast
from ._utils import check_if_whitespace_only, logger_warning
from .constants import ColorSpaces
from .errors import PdfReadError
from .generic import (
def bits2byte(data: bytes, size: Tuple[int, int], bits: int) -> bytes:
    mask = (2 << bits) - 1
    nbuff = bytearray(size[0] * size[1])
    by = 0
    bit = 8 - bits
    for y in range(size[1]):
        if bit != 0 and bit != 8 - bits:
            by += 1
            bit = 8 - bits
        for x in range(size[0]):
            nbuff[y * size[0] + x] = data[by] >> bit & mask
            bit -= bits
            if bit < 0:
                by += 1
                bit = 8 - bits
    return bytes(nbuff)