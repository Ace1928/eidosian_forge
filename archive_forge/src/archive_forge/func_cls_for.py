import collections
import functools
import inspect
import operator
import types
from typing import Dict, List, Optional
import torch
import torch.fx
from ..._guards import Source
from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, GetItemSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable
@staticmethod
def cls_for(obj):
    return {iter: ListIteratorVariable, list: ListVariable, slice: SliceVariable, torch.Size: SizeVariable, tuple: TupleVariable, odict_values: ListVariable, torch.nn.ParameterList: ListVariable, torch.nn.ModuleList: ListVariable, collections.deque: DequeVariable}[obj]