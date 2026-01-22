import logging
import re
from typing import Optional, Union
from .enums import LanguageFilter, ProbingState
@staticmethod
def filter_high_byte_only(buf: Union[bytes, bytearray]) -> bytes:
    buf = re.sub(b'([\x00-\x7f])+', b' ', buf)
    return buf