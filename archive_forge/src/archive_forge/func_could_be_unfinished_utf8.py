import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
def could_be_unfinished_utf8(seq: bytes) -> bool:
    o = ord(seq[0:1])
    return o & 224 == 192 and len(seq) < 2 or (o & 240 == 224 and len(seq) < 3) or (o & 248 == 240 and len(seq) < 4) or (o & 252 == 248 and len(seq) < 5) or (o & 254 == 252 and len(seq) < 6)