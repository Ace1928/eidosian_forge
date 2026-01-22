from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
def trace_max_mem(self, frame, event, arg):
    if event in ('line', 'return') and frame.f_code in self.code_map:
        c = _get_memory(-1, self.backend, filename=frame.f_code.co_filename)
        if c >= self.max_mem:
            t = 'Current memory {0:.2f} MiB exceeded the maximum of {1:.2f} MiB\n'.format(c, self.max_mem)
            sys.stdout.write(t)
            sys.stdout.write('Stepping into the debugger \n')
            frame.f_lineno -= 2
            p = pdb.Pdb()
            p.quitting = False
            p.stopframe = frame
            p.returnframe = None
            p.stoplineno = frame.f_lineno - 3
            p.botframe = None
            return p.trace_dispatch
    if self._original_trace_function is not None:
        self._original_trace_function(frame, event, arg)
    return self.trace_max_mem