import io
import math
import os
import typing
import weakref
def recover_line_quad(line: dict, spans: list=None) -> fitz.Quad:
    """Calculate the line quad for 'dict' / 'rawdict' text extractions.

    The lower quad points are those of the first, resp. last span quad.
    The upper points are determined by the maximum span quad height.
    From this, compute a rect with bottom-left in (0, 0), convert this to a
    quad and rotate and shift back to cover the text of the spans.

    Args:
        spans: (list, optional) sub-list of spans to consider.
    Returns:
        fitz.Quad covering selected spans.
    """
    if spans is None:
        spans = line['spans']
    if len(spans) == 0:
        raise ValueError('bad span list')
    line_dir = line['dir']
    cos, sin = line_dir
    q0 = recover_quad(line_dir, spans[0])
    if len(spans) > 1:
        q1 = recover_quad(line_dir, spans[-1])
    else:
        q1 = q0
    line_ll = q0.ll
    line_lr = q1.lr
    mat0 = fitz.planish_line(line_ll, line_lr)
    x_lr = line_lr * mat0
    small = fitz.TOOLS.set_small_glyph_heights()
    h = max([s['size'] * (1 if small else s['ascender'] - s['descender']) for s in spans])
    line_rect = fitz.Rect(0, -h, x_lr.x, 0)
    line_quad = line_rect.quad
    line_quad *= ~mat0
    return line_quad