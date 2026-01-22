import re
import string
from .util import escape_url
def parse_link_label(src, start_pos):
    m = _INLINE_LINK_LABEL_RE.match(src, start_pos)
    if m:
        label = m.group(0)[:-1]
        return (label, m.end())
    return (None, None)