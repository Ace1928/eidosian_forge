import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@functools.wraps(func)
def assert_context(*args, **kwargs):
    if not context_value.get():
        raise exceptions.MissingCallbackContextException(f'dash.callback_context.{getattr(func, '__name__')} is only available from a callback!')
    return func(*args, **kwargs)