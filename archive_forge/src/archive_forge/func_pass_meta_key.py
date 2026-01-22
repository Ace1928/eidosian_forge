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
def pass_meta_key(key: str, *, doc_description: t.Optional[str]=None) -> 't.Callable[[t.Callable[te.Concatenate[t.Any, P], R]], t.Callable[P, R]]':
    """Create a decorator that passes a key from
    :attr:`click.Context.meta` as the first argument to the decorated
    function.

    :param key: Key in ``Context.meta`` to pass.
    :param doc_description: Description of the object being passed,
        inserted into the decorator's docstring. Defaults to "the 'key'
        key from Context.meta".

    .. versionadded:: 8.0
    """

    def decorator(f: 't.Callable[te.Concatenate[t.Any, P], R]') -> 't.Callable[P, R]':

        def new_func(*args: 'P.args', **kwargs: 'P.kwargs') -> R:
            ctx = get_current_context()
            obj = ctx.meta[key]
            return ctx.invoke(f, obj, *args, **kwargs)
        return update_wrapper(new_func, f)
    if doc_description is None:
        doc_description = f'the {key!r} key from :attr:`click.Context.meta`'
    decorator.__doc__ = f'Decorator that passes {doc_description} as the first argument to the decorated function.'
    return decorator