from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
from .cells import (
from .repr import Result, rich_repr
from .style import Style
def split_cells(self, cut: int) -> Tuple['Segment', 'Segment']:
    """Split segment in to two segments at the specified column.

        If the cut point falls in the middle of a 2-cell wide character then it is replaced
        by two spaces, to preserve the display width of the parent segment.

        Returns:
            Tuple[Segment, Segment]: Two segments.
        """
    text, style, control = self
    if _is_single_cell_widths(text):
        if cut >= len(text):
            return (self, Segment('', style, control))
        return (Segment(text[:cut], style, control), Segment(text[cut:], style, control))
    return self._split_cells(self, cut)