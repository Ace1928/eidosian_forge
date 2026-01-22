import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def collate_line(line_chars: list, tolerance=DEFAULT_X_TOLERANCE) -> str:
    coll = ''
    last_x1 = None
    for char in sorted(line_chars, key=itemgetter('x0')):
        if last_x1 is not None and char['x0'] > last_x1 + tolerance:
            coll += ' '
        last_x1 = char['x1']
        coll += char['text']
    return coll