import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def runctx(self, cmd, globals=None, locals=None):
    if globals is None:
        globals = {}
    if locals is None:
        locals = {}
    if not self.donothing:
        threading.settrace(self.globaltrace)
        sys.settrace(self.globaltrace)
    try:
        exec(cmd, globals, locals)
    finally:
        if not self.donothing:
            sys.settrace(None)
            threading.settrace(None)