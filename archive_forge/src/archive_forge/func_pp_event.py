import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
def pp_event(seq: Union[Event, str]) -> Union[str, bytes]:
    """Returns pretty representation of an Event or keypress"""
    if isinstance(seq, Event):
        return str(seq)
    rev_curses = {v: k for k, v in CURSES_NAMES.items()}
    rev_curtsies = {v: k for k, v in CURTSIES_NAMES.items()}
    bytes_seq: Optional[bytes] = None
    if seq in rev_curses:
        bytes_seq = rev_curses[seq]
    elif seq in rev_curtsies:
        bytes_seq = rev_curtsies[seq]
    if bytes_seq:
        pretty = curtsies_name(bytes_seq)
        if pretty != seq:
            return pretty
    return repr(seq).lstrip('u')[1:-1]