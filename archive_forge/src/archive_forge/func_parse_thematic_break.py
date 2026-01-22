import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def parse_thematic_break(self, m: Match, state: BlockState) -> int:
    """Parse token for thematic break, e.g. ``<hr>`` tag in HTML."""
    state.append_token({'type': 'thematic_break'})
    return m.end() + 1