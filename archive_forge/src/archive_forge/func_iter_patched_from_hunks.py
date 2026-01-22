import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def iter_patched_from_hunks(orig_lines, hunks):
    """Iterate through a series of lines with a patch applied.
    This handles a single file, and does exact, not fuzzy patching.

    :param orig_lines: The unpatched lines.
    :param hunks: An iterable of Hunk instances.
    """
    seen_patch = []
    line_no = 1
    if orig_lines is not None:
        orig_lines = iter(orig_lines)
    for hunk in hunks:
        while line_no < hunk.orig_pos:
            orig_line = next(orig_lines)
            yield orig_line
            line_no += 1
        for hunk_line in hunk.lines:
            seen_patch.append(hunk_line.contents)
            if isinstance(hunk_line, InsertLine):
                yield hunk_line.contents
            elif isinstance(hunk_line, (ContextLine, RemoveLine)):
                orig_line = next(orig_lines)
                if orig_line != hunk_line.contents:
                    raise PatchConflict(line_no, orig_line, b''.join(seen_patch))
                if isinstance(hunk_line, ContextLine):
                    yield orig_line
                elif not isinstance(hunk_line, RemoveLine):
                    raise AssertionError(hunk_line)
                line_no += 1
    if orig_lines is not None:
        yield from orig_lines