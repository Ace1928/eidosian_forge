import io
import math
import os
import typing
import weakref
def get_text_selection(page: fitz.Page, p1: point_like, p2: point_like, clip: rect_like=None, textpage: fitz.TextPage=None):
    fitz.CheckParent(page)
    tp = textpage
    if tp is None:
        tp = page.get_textpage(clip=clip, flags=fitz.TEXT_DEHYPHENATE)
    elif getattr(tp, 'parent') != page:
        raise ValueError('not a textpage of this page')
    rc = tp.extractSelection(p1, p2)
    if textpage is None:
        del tp
    return rc