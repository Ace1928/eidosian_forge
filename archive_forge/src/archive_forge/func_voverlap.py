import heapq
import logging
from typing import (
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdfinterp import Color
from .pdfinterp import PDFGraphicState
from .pdftypes import PDFStream
from .utils import INF, PathSegment
from .utils import LTComponentT
from .utils import Matrix
from .utils import Plane
from .utils import Point
from .utils import Rect
from .utils import apply_matrix_pt
from .utils import bbox2str
from .utils import fsplit
from .utils import get_bound
from .utils import matrix2str
from .utils import uniq
def voverlap(self, obj: 'LTComponent') -> float:
    assert isinstance(obj, LTComponent), str(type(obj))
    if self.is_voverlap(obj):
        return min(abs(self.y0 - obj.y1), abs(self.y1 - obj.y0))
    else:
        return 0