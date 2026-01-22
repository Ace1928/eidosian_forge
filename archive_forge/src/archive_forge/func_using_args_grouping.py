import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def using_args_grouping(self):
    """
        Return True if this callback is using dictionary or nested groupings for
        Input/State dependencies, or if Input and State dependencies are interleaved
        """
    return getattr(_get_context_value(), 'using_args_grouping', [])