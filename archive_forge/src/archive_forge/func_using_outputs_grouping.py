import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def using_outputs_grouping(self):
    """
        Return True if this callback is using dictionary or nested groupings for
        Output dependencies.
        """
    return getattr(_get_context_value(), 'using_outputs_grouping', [])