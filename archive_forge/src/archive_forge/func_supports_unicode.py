from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def supports_unicode() -> bool:
    """
    Return True if python is able to convert non-ascii unicode strings
    to the current encoding.
    """
    return _target_encoding and _target_encoding != 'ascii'