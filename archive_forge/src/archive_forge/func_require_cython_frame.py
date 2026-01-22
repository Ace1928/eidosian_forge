from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def require_cython_frame(function):

    @functools.wraps(function)
    @require_running_program
    def wrapper(self, *args, **kwargs):
        frame = kwargs.get('frame') or gdb.selected_frame()
        if not self.is_cython_function(frame):
            raise gdb.GdbError('Selected frame does not correspond with a Cython function we know about.')
        return function(self, *args, **kwargs)
    return wrapper