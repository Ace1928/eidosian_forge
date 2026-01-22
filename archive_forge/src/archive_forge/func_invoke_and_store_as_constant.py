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
def invoke_and_store_as_constant(tx, fn, name, args, kwargs):

    def convert(x):
        if isinstance(x, variables.TensorVariable):
            return x.get_real_value()
        return x.as_python_constant()
    args = [convert(x) for x in args]
    kwargs = {k: convert(v) for k, v in kwargs.items()}
    res = fn(*args, **kwargs)
    return tx.output.register_attr_or_module(res, name, source=ConstantSource(name))