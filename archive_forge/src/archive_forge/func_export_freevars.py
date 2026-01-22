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
def export_freevars(self, parent, child):
    code = self.get_code()
    for var in code.co_freevars:
        if var in child.symbolic_locals:
            parent.symbolic_locals[var] = child.symbolic_locals[var]