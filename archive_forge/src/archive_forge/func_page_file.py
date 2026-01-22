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
def page_file(fname, start=0, pager_cmd=None):
    """Page a file, using an optional pager command and starting line.
    """
    pager_cmd = get_pager_cmd(pager_cmd)
    pager_cmd += ' ' + get_pager_start(pager_cmd, start)
    try:
        if os.environ['TERM'] in ['emacs', 'dumb']:
            raise EnvironmentError
        system(pager_cmd + ' ' + fname)
    except:
        try:
            if start > 0:
                start -= 1
            page(open(fname, encoding='utf-8').read(), start)
        except:
            print('Unable to show file', repr(fname))