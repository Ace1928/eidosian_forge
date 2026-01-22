import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def sanitized_lines(lines: Collection[Tuple[int, int]], src_contents: str) -> Collection[Tuple[int, int]]:
    """Returns the valid line ranges for the given source.

    This removes ranges that are entirely outside the valid lines.

    Other ranges are normalized so that the start values are at least 1 and the
    end values are at most the (1-based) index of the last source line.
    """
    if not src_contents:
        return []
    good_lines = []
    src_line_count = src_contents.count('\n')
    if not src_contents.endswith('\n'):
        src_line_count += 1
    for start, end in lines:
        if start > src_line_count:
            continue
        start = max(start, 1)
        if end < start:
            continue
        end = min(end, src_line_count)
        good_lines.append((start, end))
    return good_lines