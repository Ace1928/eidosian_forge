from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def pointervalue(gdbval):
    pointer = _pointervalue(gdbval)
    try:
        if pointer < 0:
            raise gdb.GdbError('Negative pointer value, presumably a bug in gdb, aborting.')
    except RuntimeError:
        pass
    return pointer