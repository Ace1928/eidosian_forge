import re
from functools import partial, reduce
from math import gcd
from operator import itemgetter
from typing import (
from ._loop import loop_last
from ._pick import pick_bool
from ._wrap import divide_line
from .align import AlignMethod
from .cells import cell_len, set_cell_size
from .containers import Lines
from .control import strip_control_codes
from .emoji import EmojiVariant
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import Style, StyleType
def right_crop(self, amount: int=1) -> None:
    """Remove a number of characters from the end of the text."""
    max_offset = len(self.plain) - amount
    _Span = Span
    self._spans[:] = [span if span.end < max_offset else _Span(span.start, min(max_offset, span.end), span.style) for span in self._spans if span.start < max_offset]
    self._text = [self.plain[:-amount]]
    self._length -= amount