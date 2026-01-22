from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def decompose_tagmarkup(tm):
    """Return (text string, attribute list) for tagmarkup passed."""
    tl, al = _tagmarkup_recurse(tm, None)
    if tl:
        text = tl[0][:0].join(tl)
    else:
        text = ''
    if al and al[-1][0] is None:
        del al[-1]
    return (text, al)