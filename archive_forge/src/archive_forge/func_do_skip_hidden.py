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
def do_skip_hidden(self, arg):
    """
        Change whether or not we should skip frames with the
        __tracebackhide__ attribute.
        """
    if not arg.strip():
        print(f"skip_hidden = {self.skip_hidden}, use 'yes','no', 'true', or 'false' to change.")
    elif arg.strip().lower() in ('true', 'yes'):
        self.skip_hidden = True
    elif arg.strip().lower() in ('false', 'no'):
        self.skip_hidden = False
    if not any(self._predicates.values()):
        print('Warning, all predicates set to False, skip_hidden may not have any effects.')