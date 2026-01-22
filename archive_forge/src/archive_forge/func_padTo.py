from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def padTo(n, seq, default=None):
    """
    Pads a sequence out to n elements,

    filling in with a default value if it is not long enough.

    If the input sequence is longer than n, raises ValueError.

    Details, details:
    This returns a new list; it does not extend the original sequence.
    The new list contains the values of the original sequence, not copies.
    """
    if len(seq) > n:
        raise ValueError('%d elements is more than %d.' % (len(seq), n))
    blank = [default] * n
    blank[:len(seq)] = list(seq)
    return blank