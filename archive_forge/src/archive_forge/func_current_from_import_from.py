import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def current_from_import_from(cursor_offset: int, line: str) -> Optional[LinePart]:
    """If in from import completion, the word after from

    returns None if cursor not in or just after one of the two interesting
    parts of an import: from (module) import (name1, name2)
    """
    for m in _current_from_import_from_re.finditer(line):
        if m.start(1) < cursor_offset <= m.end(1) or m.start(2) < cursor_offset <= m.end(2):
            return LinePart(m.start(1), m.end(1), m.group(1))
    return None