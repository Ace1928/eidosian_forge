import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def shift_to_mod_lines(self, pos):
    position = self.orig_pos - 1
    shift = 0
    for line in self.lines:
        if isinstance(line, InsertLine):
            shift += 1
        elif isinstance(line, RemoveLine):
            if position == pos:
                return None
            shift -= 1
            position += 1
        elif isinstance(line, ContextLine):
            position += 1
        if position > pos:
            break
    return shift