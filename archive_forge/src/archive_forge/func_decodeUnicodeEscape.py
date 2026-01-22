import codecs
import re
import warnings
from typing import Match
def decodeUnicodeEscape(escaped: str) -> str:
    if '\\' not in escaped:
        return escaped
    return _turtle_escape_pattern.sub(_turtle_escape_subber, escaped)