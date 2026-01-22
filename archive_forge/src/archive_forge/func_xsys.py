import builtins as builtin_mod
import sys
import types
from pathlib import Path
from . import tools
from IPython.core import page
from IPython.utils import io
from IPython.terminal.interactiveshell import TerminalInteractiveShell
def xsys(self, cmd):
    """Replace the default system call with a capturing one for doctest.
    """
    print(self.getoutput(cmd, split=False, depth=1).rstrip(), end='', file=sys.stdout)
    sys.stdout.flush()