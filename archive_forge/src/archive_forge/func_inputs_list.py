import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def inputs_list(self):
    if self.using_args_grouping:
        warnings.warn('inputs_list is deprecated, use args_grouping instead', DeprecationWarning)
    return getattr(_get_context_value(), 'inputs_list', [])