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
def globaltrace_trackcallers(self, frame, why, arg):
    """Handler for call events.

        Adds information about who called who to the self._callers dict.
        """
    if why == 'call':
        this_func = self.file_module_function_of(frame)
        parent_func = self.file_module_function_of(frame.f_back)
        self._callers[parent_func, this_func] = 1