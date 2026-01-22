import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def get_pfunctions(self):
    p_functions = []
    for name, item in self.pdict.items():
        if not name.startswith('p_') or name == 'p_error':
            continue
        if isinstance(item, (types.FunctionType, types.MethodType)):
            line = getattr(item, 'co_firstlineno', item.__code__.co_firstlineno)
            module = inspect.getmodule(item)
            p_functions.append((line, module, name, item.__doc__))
    p_functions.sort(key=lambda p_function: (p_function[0], str(p_function[1]), p_function[2], p_function[3]))
    self.pfuncs = p_functions