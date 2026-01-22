import sys
from collections.abc import MutableSequence
import re
from textwrap import dedent
from keyword import iskeyword
import flask
from ._grouping import grouping_len, map_grouping
from .development.base_component import Component
from . import exceptions
from ._utils import (
def validate_and_group_input_args(flat_args, arg_index_grouping):
    if grouping_len(arg_index_grouping) != len(flat_args):
        raise exceptions.CallbackException('Inputs do not match callback definition')
    args_grouping = map_grouping(lambda ind: flat_args[ind], arg_index_grouping)
    if isinstance(arg_index_grouping, dict):
        func_args = []
        func_kwargs = args_grouping
        for key in func_kwargs:
            if not key.isidentifier():
                raise exceptions.CallbackException(f'{key} is not a valid Python variable name')
    elif isinstance(arg_index_grouping, (tuple, list)):
        func_args = list(args_grouping)
        func_kwargs = {}
    else:
        func_args = [args_grouping]
        func_kwargs = {}
    return (func_args, func_kwargs)