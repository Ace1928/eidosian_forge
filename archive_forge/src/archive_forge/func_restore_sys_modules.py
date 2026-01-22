from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def restore_sys_modules(scrubbed):
    """
    Add any previously scrubbed modules back to the sys.modules cache,
    but only if it's safe to do so.
    """
    clash = set(sys.modules) & set(scrubbed)
    if len(clash) != 0:
        first = list(clash)[0]
        raise ImportError('future module {} clashes with Py2 module'.format(first))
    sys.modules.update(scrubbed)