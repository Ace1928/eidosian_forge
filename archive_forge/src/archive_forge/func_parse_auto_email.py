import re
from typing import Optional, List, Dict, Any, Match
from .core import Parser, InlineState
from .util import (
from .helpers import (
def parse_auto_email(self, m: Match, state: InlineState) -> int:
    text = m.group(0)
    pos = m.end()
    if state.in_link:
        self.process_text(text, state)
        return pos
    text = text[1:-1]
    url = 'mailto:' + text
    self._add_auto_link(url, text, state)
    return pos