import inspect
import types
import typing as t
from functools import update_wrapper
from gettext import gettext as _
from .core import Argument
from .core import Command
from .core import Context
from .core import Group
from .core import Option
from .core import Parameter
from .globals import get_current_context
from .utils import echo
def pass_obj(f: 't.Callable[te.Concatenate[t.Any, P], R]') -> 't.Callable[P, R]':
    """Similar to :func:`pass_context`, but only pass the object on the
    context onwards (:attr:`Context.obj`).  This is useful if that object
    represents the state of a nested system.
    """

    def new_func(*args: 'P.args', **kwargs: 'P.kwargs') -> 'R':
        return f(get_current_context().obj, *args, **kwargs)
    return update_wrapper(new_func, f)