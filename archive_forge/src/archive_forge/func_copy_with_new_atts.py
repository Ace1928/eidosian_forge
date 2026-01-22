import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def copy_with_new_atts(self, **attributes: Union[bool, int]) -> 'FmtStr':
    """Returns a new FmtStr with the same content but new formatting"""
    return FmtStr(*(Chunk(bfs.s, bfs.atts.extend(attributes)) for bfs in self.chunks))