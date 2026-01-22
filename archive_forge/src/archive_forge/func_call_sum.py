import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
from collections import defaultdict, OrderedDict
from typing import Dict, List
import torch
from torch import sym_float, sym_int
from .. import config, polyfill, variables
from ..exc import (
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import AttrSource, GetItemSource, is_constant_source, TypeSource
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import ConstDictVariable, DefaultDictVariable, SetVariable
from .lists import (
from .tensor import FakeItemVariable, SymNodeVariable, UnspecializedPythonVariable
from .user_defined import UserDefinedVariable
def call_sum(self, tx, seq, **kwargs):
    if isinstance(seq, (variables.ListVariable, variables.TupleVariable)) and all((isinstance(x, variables.ConstantVariable) and isinstance(x.value, (int, float)) for x in seq.items)) and (not kwargs):
        new_list = [x.value for x in seq.items]
        return variables.ConstantVariable.create(sum(new_list))
    if seq.has_unpack_var_sequence(tx):
        start = kwargs.pop('start', variables.ConstantVariable.create(0)).as_python_constant()
        assert not kwargs
        items = seq.unpack_var_sequence(tx)[start:]
        return BuiltinVariable(functools.reduce).call_function(tx, [BuiltinVariable(operator.add), variables.TupleVariable(items), variables.ConstantVariable.create(0)], {})