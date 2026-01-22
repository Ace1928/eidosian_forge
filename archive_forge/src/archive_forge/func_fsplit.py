import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def fsplit(pred: Callable[[_T], bool], objs: Iterable[_T]) -> Tuple[List[_T], List[_T]]:
    """Split a list into two classes according to the predicate."""
    t = []
    f = []
    for obj in objs:
        if pred(obj):
            t.append(obj)
        else:
            f.append(obj)
    return (t, f)