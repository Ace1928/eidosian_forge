from dataclasses import dataclass
from typing import List
from .align import get_alignments
from .alignment_array import AlignmentArray
@classmethod
def from_indices(cls, x2y: List[List[int]], y2x: List[List[int]]) -> 'Alignment':
    x2y = AlignmentArray(x2y)
    y2x = AlignmentArray(y2x)
    return Alignment(x2y=x2y, y2x=y2x)