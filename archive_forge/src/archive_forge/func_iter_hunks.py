import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def iter_hunks(iter_lines, allow_dirty=False):
    """
    :arg iter_lines: iterable of lines to parse for hunks
    :kwarg allow_dirty: If True, when we encounter something that is not
        a hunk header when we're looking for one, assume the rest of the lines
        are not part of the patch (comments or other junk).  Default False
    """
    hunk = None
    for line in iter_lines:
        if line == b'\n':
            if hunk is not None:
                yield hunk
                hunk = None
            continue
        if hunk is not None:
            yield hunk
        try:
            hunk = hunk_from_header(line)
        except MalformedHunkHeader:
            if allow_dirty:
                return
            raise
        orig_size = 0
        mod_size = 0
        while orig_size < hunk.orig_range or mod_size < hunk.mod_range:
            hunk_line = parse_line(next(iter_lines))
            hunk.lines.append(hunk_line)
            if isinstance(hunk_line, (RemoveLine, ContextLine)):
                orig_size += 1
            if isinstance(hunk_line, (InsertLine, ContextLine)):
                mod_size += 1
    if hunk is not None:
        yield hunk