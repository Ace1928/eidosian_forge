import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def go(*args, **kw):
    return dec(context, *args, **kw)