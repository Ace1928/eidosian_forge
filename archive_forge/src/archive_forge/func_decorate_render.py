import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def decorate_render(render_fn):
    dec = fn(render_fn)

    def go(*args, **kw):
        return dec(context, *args, **kw)
    return go