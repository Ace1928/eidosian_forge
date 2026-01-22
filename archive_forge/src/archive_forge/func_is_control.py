from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
from .cells import (
from .repr import Result, rich_repr
from .style import Style
@property
def is_control(self) -> bool:
    """Check if the segment contains control codes."""
    return self.control is not None