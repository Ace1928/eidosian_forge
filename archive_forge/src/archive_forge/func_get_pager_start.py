import os
import io
import re
import sys
import tempfile
import subprocess
from io import UnsupportedOperation
from pathlib import Path
from IPython import get_ipython
from IPython.display import display
from IPython.core.error import TryNext
from IPython.utils.data import chop
from IPython.utils.process import system
from IPython.utils.terminal import get_terminal_size
from IPython.utils import py3compat
def get_pager_start(pager, start):
    """Return the string for paging files with an offset.

    This is the '+N' argument which less and more (under Unix) accept.
    """
    if pager in ['less', 'more']:
        if start:
            start_string = '+' + str(start)
        else:
            start_string = ''
    else:
        start_string = ''
    return start_string