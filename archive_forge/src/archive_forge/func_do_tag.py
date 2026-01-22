from typing import (
from pdfminer.psparser import PSLiteral
from . import utils
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import Matrix, Point, Rect, PathSegment
def do_tag(self, tag: PSLiteral, props: Optional['PDFStackT']=None) -> None:
    self.begin_tag(tag, props)
    self._stack.pop(-1)
    return