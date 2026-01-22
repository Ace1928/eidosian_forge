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
def do_until(self, arg):
    """unt(il) [lineno]
        Without argument, continue execution until the line with a
        number greater than the current one is reached.  With a line
        number, continue execution until a line with a number greater
        or equal to that is reached.  In both cases, also stop when
        the current frame returns.
        """
    if arg:
        try:
            lineno = int(arg)
        except ValueError:
            self.error('Error in argument: %r' % arg)
            return
        if lineno <= self.curframe.f_lineno:
            self.error('"until" line number is smaller than current line number')
            return
    else:
        lineno = None
    self.set_until(self.curframe, lineno)
    return 1