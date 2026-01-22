import re
import string
from .util import escape_url
def parse_link_href(src, start_pos, block=False):
    m = LINK_BRACKET_START.match(src, start_pos)
    if m:
        start_pos = m.end() - 1
        m = LINK_BRACKET_RE.match(src, start_pos)
        if m:
            return (m.group(1), m.end())
        return (None, None)
    if block:
        m = LINK_HREF_BLOCK_RE.match(src, start_pos)
    else:
        m = LINK_HREF_INLINE_RE.match(src, start_pos)
    if not m:
        return (None, None)
    end_pos = m.end()
    href = m.group(1)
    if block and src[end_pos - 1] == href[-1]:
        return (href, end_pos)
    return (href, end_pos - 1)