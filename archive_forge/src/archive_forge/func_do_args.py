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
def do_args(self, arg):
    """a(rgs)
        Print the argument list of the current function.
        """
    co = self.curframe.f_code
    dict = self.curframe_locals
    n = co.co_argcount + co.co_kwonlyargcount
    if co.co_flags & inspect.CO_VARARGS:
        n = n + 1
    if co.co_flags & inspect.CO_VARKEYWORDS:
        n = n + 1
    for i in range(n):
        name = co.co_varnames[i]
        if name in dict:
            self.message('%s = %s' % (name, self._safe_repr(dict[name], name)))
        else:
            self.message('%s = *** undefined ***' % (name,))