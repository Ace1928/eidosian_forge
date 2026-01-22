from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
def short_stack(skip: int=0, full: bool=False, frame_ids: bool=False, short_filenames: bool=False) -> str:
    """Return a string summarizing the call stack.

    The string is multi-line, with one line per stack frame. Each line shows
    the function name, the file name, and the line number:

        ...
        start_import_stop : /Users/ned/coverage/trunk/tests/coveragetest.py:95
        import_local_file : /Users/ned/coverage/trunk/tests/coveragetest.py:81
        import_local_file : /Users/ned/coverage/trunk/coverage/backward.py:159
        ...

    `skip` is the number of closest immediate frames to skip, so that debugging
    functions can call this and not be included in the result.

    If `full` is true, then include all frames.  Otherwise, initial "boring"
    frames (ones in site-packages and earlier) are omitted.

    `short_filenames` will shorten filenames using `short_filename`, to reduce
    the amount of repetitive noise in stack traces.

    """
    BORING_PRELUDE = ['<string>', '\\bigor.py$', '\\bsite-packages\\b']
    stack: Iterable[inspect.FrameInfo] = inspect.stack()[:skip:-1]
    if not full:
        for pat in BORING_PRELUDE:
            stack = itertools.dropwhile(lambda fi, pat=pat: re.search(pat, fi.filename), stack)
    lines = []
    for frame_info in stack:
        line = f'{frame_info.function:>30s} : '
        if frame_ids:
            line += f'{id(frame_info.frame):#x} '
        filename = frame_info.filename
        if short_filenames:
            filename = short_filename(filename)
        line += f'{filename}:{frame_info.lineno}'
        lines.append(line)
    return '\n'.join(lines)