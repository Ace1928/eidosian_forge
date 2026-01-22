from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def mutter_callsite(stacklevel, fmt, *args):
    """Perform a mutter of fmt and args, logging the call trace.

    :param stacklevel: The number of frames to show. None will show all
        frames.
    :param fmt: The format string to pass to mutter.
    :param args: A list of substitution variables.
    """
    import traceback
    outf = StringIO()
    if stacklevel is None:
        limit = None
    else:
        limit = stacklevel + 1
    traceback.print_stack(limit=limit, file=outf)
    formatted_lines = outf.getvalue().splitlines()
    formatted_stack = '\n'.join(formatted_lines[:-2])
    mutter(fmt + '\nCalled from:\n%s', *args + (formatted_stack,))