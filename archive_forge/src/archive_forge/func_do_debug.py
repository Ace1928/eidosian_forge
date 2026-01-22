import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def do_debug(self, arg):
    """debug code
        Enter a recursive debugger that steps through the code
        argument (which is an arbitrary expression or statement to be
        executed in the current environment).
        """
    sys.settrace(None)
    globals = self.curframe.f_globals
    locals = self.curframe_locals
    p = Pdb(self.completekey, self.stdin, self.stdout)
    p.prompt = '(%s) ' % self.prompt.strip()
    self.message('ENTERING RECURSIVE DEBUGGER')
    try:
        sys.call_tracing(p.run, (arg, globals, locals))
    except Exception:
        self._error_exc()
    self.message('LEAVING RECURSIVE DEBUGGER')
    sys.settrace(self.trace_dispatch)
    self.lastcmd = p.lastcmd