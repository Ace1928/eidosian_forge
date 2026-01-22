import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def make_arrow(pad):
    """generate the leading arrow in front of traceback or debugger"""
    if pad >= 2:
        return '-' * (pad - 2) + '> '
    elif pad == 1:
        return '>'
    return ''