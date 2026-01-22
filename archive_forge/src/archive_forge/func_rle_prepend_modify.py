from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def rle_prepend_modify(rle, a_r) -> None:
    """
    Append (a, r) (unpacked from *a_r*) to BEGINNING of rle.
    Merge with first run when possible

    MODIFIES rle parameter contents. Returns None.
    """
    a, r = a_r
    if not rle:
        rle[:] = [(a, r)]
    else:
        al, run = rle[0]
        if a == al:
            rle[0] = (a, run + r)
        else:
            rle[0:0] = [(a, r)]