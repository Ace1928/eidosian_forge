import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def parse_axt_heading(self, m: Match, state: BlockState) -> int:
    """Parse token for AXT heading. An AXT heading is started with 1 to 6
        symbol of ``#``."""
    level = len(m.group('axt_1'))
    text = m.group('axt_2').strip()
    if text:
        text = _AXT_HEADING_TRIM.sub('', text)
    token = {'type': 'heading', 'text': text, 'attrs': {'level': level}, 'style': 'axt'}
    state.append_token(token)
    return m.end() + 1