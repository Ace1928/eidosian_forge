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
def do_pinfo2(self, arg):
    """Provide extra detailed information about an object.

        The debugger interface to %pinfo2, i.e., obj??."""
    namespaces = [('Locals', self.curframe_locals), ('Globals', self.curframe.f_globals)]
    self.shell.find_line_magic('pinfo2')(arg, namespaces=namespaces)