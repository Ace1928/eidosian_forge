import re
from ast import literal_eval
from operator import attrgetter
from typing import Callable, Iterable, List, Match, NamedTuple, Optional, Tuple, Union
from ._emoji_replace import _emoji_replace
from .emoji import EmojiVariant
from .errors import MarkupError
from .style import Style
from .text import Span, Text
def escape_backslashes(match: Match[str]) -> str:
    """Called by re.sub replace matches."""
    backslashes, text = match.groups()
    return f'{backslashes}{backslashes}\\{text}'