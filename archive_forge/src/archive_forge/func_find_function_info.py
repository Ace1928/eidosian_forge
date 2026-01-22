import logging
import sys
import types
import threading
import inspect
from functools import wraps
from itertools import chain
from numba.core import config
def find_function_info(func, spec, args):
    """Return function meta-data in a tuple.

    (name, type)"""
    module = getattr(func, '__module__', None)
    name = getattr(func, '__name__', None)
    self = getattr(func, '__self__', None)
    cname = None
    if self:
        cname = self.__name__
    elif len(spec.args) and spec.args[0] == 'self':
        cname = args[0].__class__.__name__
    elif len(spec.args) and spec.args[0] == 'cls':
        cname = args[0].__name__
    if name:
        qname = []
        if module and module != '__main__':
            qname.append(module)
            qname.append('.')
        if cname:
            qname.append(cname)
            qname.append('.')
        qname.append(name)
        name = ''.join(qname)
    return (name, None)