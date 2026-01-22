import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def new_with_atts_removed(self, *attributes: str) -> 'FmtStr':
    """Returns a new FmtStr with the same content but some attributes removed"""
    result = FmtStr(*(Chunk(bfs.s, bfs.atts.remove(*attributes)) for bfs in self.chunks))
    return result