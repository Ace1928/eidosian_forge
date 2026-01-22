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
def new_do_restart(self, arg):
    """Restart command. In the context of ipython this is exactly the same
        thing as 'quit'."""
    self.msg("Restart doesn't make sense here. Using 'quit' instead.")
    return self.do_quit(arg)