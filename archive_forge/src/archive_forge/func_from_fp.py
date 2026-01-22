import logging
from os.path import basename, splitext
from typing import BinaryIO, List, Optional, Set
from .cd import (
from .constant import IANA_SUPPORTED, TOO_BIG_SEQUENCE, TOO_SMALL_SEQUENCE, TRACE
from .md import mess_ratio
from .models import CharsetMatch, CharsetMatches
from .utils import (
def from_fp(fp: BinaryIO, steps: int=5, chunk_size: int=512, threshold: float=0.2, cp_isolation: List[str]=None, cp_exclusion: List[str]=None, preemptive_behaviour: bool=True, explain: bool=False) -> CharsetMatches:
    """
    Same thing than the function from_bytes but using a file pointer that is already ready.
    Will not close the file pointer.
    """
    return from_bytes(fp.read(), steps, chunk_size, threshold, cp_isolation, cp_exclusion, preemptive_behaviour, explain)