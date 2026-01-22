import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def iter_lines_handle_nl(iter_lines: Iterator[bytes]) -> Iterator[bytes]:
    """
    Iterates through lines, ensuring that lines that originally had no
    terminating 
 are produced without one.  This transformation may be
    applied at any point up until hunk line parsing, and is safe to apply
    repeatedly.
    """
    last_line: Optional[bytes] = None
    line: Optional[bytes]
    for line in iter_lines:
        if line == NO_NL:
            if last_line is None or not last_line.endswith(b'\n'):
                raise AssertionError()
            last_line = last_line[:-1]
            line = None
        if last_line is not None:
            yield last_line
        last_line = line
    if last_line is not None:
        yield last_line