import atexit
import os
import sys
import tempfile
from pathlib import Path
from warnings import warn
from IPython.utils.decorators import undoc
from .capture import CapturedIO, capture_output
@undoc
def raw_print_err(*args, **kw):
    """DEPRECATED: Raw print to sys.__stderr__, otherwise identical interface to print()."""
    warn('IPython.utils.io.raw_print_err has been deprecated since IPython 7.0', DeprecationWarning, stacklevel=2)
    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'), file=sys.__stderr__)
    sys.__stderr__.flush()