from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
from .cells import (
from .repr import Result, rich_repr
from .style import Style
@classmethod
def filter_control(cls, segments: Iterable['Segment'], is_control: bool=False) -> Iterable['Segment']:
    """Filter segments by ``is_control`` attribute.

        Args:
            segments (Iterable[Segment]): An iterable of Segment instances.
            is_control (bool, optional): is_control flag to match in search.

        Returns:
            Iterable[Segment]: And iterable of Segment instances.

        """
    if is_control:
        return filter(attrgetter('control'), segments)
    else:
        return filterfalse(attrgetter('control'), segments)