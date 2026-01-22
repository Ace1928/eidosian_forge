import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
def wrap_args_kwargs(tx, result):
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            result[k] = wrap_bound_arg(tx, v)