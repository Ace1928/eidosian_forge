import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def validate_error_func(self):
    if self.error_func:
        if isinstance(self.error_func, types.FunctionType):
            ismethod = 0
        elif isinstance(self.error_func, types.MethodType):
            ismethod = 1
        else:
            self.log.error("'p_error' defined, but is not a function or method")
            self.error = True
            return
        eline = self.error_func.__code__.co_firstlineno
        efile = self.error_func.__code__.co_filename
        module = inspect.getmodule(self.error_func)
        self.modules.add(module)
        argcount = self.error_func.__code__.co_argcount - ismethod
        if argcount != 1:
            self.log.error('%s:%d: p_error() requires 1 argument', efile, eline)
            self.error = True