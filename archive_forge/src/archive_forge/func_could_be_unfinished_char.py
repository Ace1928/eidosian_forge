import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
def could_be_unfinished_char(seq: bytes, encoding: str) -> bool:
    """Whether seq bytes might create a char in encoding if more bytes were added"""
    if decodable(seq, encoding):
        return False
    if codecs.getdecoder('utf8') is codecs.getdecoder(encoding):
        return could_be_unfinished_utf8(seq)
    elif codecs.getdecoder('ascii') is codecs.getdecoder(encoding):
        return False
    else:
        return True