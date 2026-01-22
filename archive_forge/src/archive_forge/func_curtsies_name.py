import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
def curtsies_name(seq: bytes) -> Union[str, bytes]:
    return CURTSIES_NAMES.get(seq, seq)