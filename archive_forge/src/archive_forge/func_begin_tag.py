from typing import (
from pdfminer.psparser import PSLiteral
from . import utils
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import Matrix, Point, Rect, PathSegment
def begin_tag(self, tag: PSLiteral, props: Optional['PDFStackT']=None) -> None:
    s = ''
    if isinstance(props, dict):
        s = ''.join([' {}="{}"'.format(utils.enc(k), utils.make_compat_str(v)) for k, v in sorted(props.items())])
    out_s = '<{}{}>'.format(utils.enc(cast(str, tag.name)), s)
    self._write(out_s)
    self._stack.append(tag)
    return