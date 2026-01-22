import re
import string
from .util import escape_url
def parse_link_text(src, pos):
    level = 1
    found = False
    start_pos = pos
    while pos < len(src):
        m = _INLINE_SQUARE_BRACKET_RE.search(src, pos)
        if not m:
            break
        pos = m.end()
        marker = m.group(0)
        if marker == ']':
            level -= 1
            if level == 0:
                found = True
                break
        else:
            level += 1
    if found:
        text = src[start_pos:pos - 1]
        return (text, pos)
    return (None, None)