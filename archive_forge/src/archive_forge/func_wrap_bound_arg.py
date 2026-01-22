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
def wrap_bound_arg(tx, val, source=None):
    if isinstance(val, VariableTracker):
        return val
    elif not source:
        from torch._dynamo.variables.builder import SourcelessBuilder
        return SourcelessBuilder()(tx, val)
    else:
        from torch._dynamo.variables.builder import VariableBuilder
        return VariableBuilder(tx, source=source)(val)