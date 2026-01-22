import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def outputs_grouping(self):
    return getattr(_get_context_value(), 'outputs_grouping', [])