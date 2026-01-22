from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
def globs_to_regex(patterns: Iterable[str], case_insensitive: bool=False, partial: bool=False) -> re.Pattern[str]:
    """Convert glob patterns to a compiled regex that matches any of them.

    Slashes are always converted to match either slash or backslash, for
    Windows support, even when running elsewhere.

    If the pattern has no slash or backslash, then it is interpreted as
    matching a file name anywhere it appears in the tree.  Otherwise, the glob
    pattern must match the whole file path.

    If `partial` is true, then the pattern will match if the target string
    starts with the pattern. Otherwise, it must match the entire string.

    Returns: a compiled regex object.  Use the .match method to compare target
    strings.

    """
    flags = 0
    if case_insensitive:
        flags |= re.IGNORECASE
    rx = join_regex(map(_glob_to_regex, patterns))
    if not partial:
        rx = f'(?:{rx})\\Z'
    compiled = re.compile(rx, flags=flags)
    return compiled