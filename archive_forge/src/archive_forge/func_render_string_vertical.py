from typing import (
from pdfminer.psparser import PSLiteral
from . import utils
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import Matrix, Point, Rect, PathSegment
def render_string_vertical(self, seq: PDFTextSeq, matrix: Matrix, pos: Point, font: PDFFont, fontsize: float, scaling: float, charspace: float, wordspace: float, rise: float, dxscale: float, ncs: PDFColorSpace, graphicstate: 'PDFGraphicState') -> Point:
    x, y = pos
    needcharspace = False
    for obj in seq:
        if isinstance(obj, (int, float)):
            y -= obj * dxscale
            needcharspace = True
        else:
            for cid in font.decode(obj):
                if needcharspace:
                    y += charspace
                y += self.render_char(utils.translate_matrix(matrix, (x, y)), font, fontsize, scaling, rise, cid, ncs, graphicstate)
                if cid == 32 and wordspace:
                    y += wordspace
                needcharspace = True
    return (x, y)