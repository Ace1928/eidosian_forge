import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def hunk_from_header(line):
    import re
    matches = re.match(b'\\@\\@ ([^@]*) \\@\\@( (.*))?\\n', line)
    if matches is None:
        raise MalformedHunkHeader('Does not match format.', line)
    try:
        orig, mod = matches.group(1).split(b' ')
    except (ValueError, IndexError) as e:
        raise MalformedHunkHeader(str(e), line)
    if not orig.startswith(b'-') or not mod.startswith(b'+'):
        raise MalformedHunkHeader("Positions don't start with + or -.", line)
    try:
        orig_pos, orig_range = parse_range(orig[1:])
        mod_pos, mod_range = parse_range(mod[1:])
    except (ValueError, IndexError) as e:
        raise MalformedHunkHeader(str(e), line)
    if mod_range < 0 or orig_range < 0:
        raise MalformedHunkHeader('Hunk range is negative', line)
    tail = matches.group(3)
    return Hunk(orig_pos, orig_range, mod_pos, mod_range, tail)