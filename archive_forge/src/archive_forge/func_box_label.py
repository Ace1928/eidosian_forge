from __future__ import annotations
import os
import re
from functools import partial
from dask.core import get_dependencies, ishashable, istask
from dask.utils import apply, funcname, import_required, key_split
def box_label(key, verbose=False):
    """Label boxes in graph by chunk index

    >>> box_label(('x', 1, 2, 3))
    '(1, 2, 3)'
    >>> box_label(('x', 123))
    '123'
    >>> box_label('x')
    ''
    """
    if isinstance(key, tuple):
        key = key[1:]
        if len(key) == 1:
            [key] = key
        return str(key)
    elif verbose:
        return str(key)
    else:
        return ''