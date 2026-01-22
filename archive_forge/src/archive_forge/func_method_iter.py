from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def method_iter():
    for func in ('get', 'get_one', 'get_all', 'new', 'edit', 'get_delete'):
        if self._find_controller(func):
            yield 'GET'
            break
    for method in ('HEAD', 'POST', 'PUT', 'DELETE', 'TRACE', 'PATCH'):
        func = method.lower()
        if self._find_controller(func):
            yield method