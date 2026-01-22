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
def call_sorted(self, tx, obj: VariableTracker, **kwargs):
    if obj.has_unpack_var_sequence(tx) and (not isinstance(obj, variables.TensorVariable)) and all((x.is_python_constant() for x in obj.unpack_var_sequence(tx))):
        function = kwargs.pop('key', None)
        reverse = kwargs.pop('reverse', ConstantVariable.create(False)).as_python_constant()
        assert len(kwargs) == 0
        if function:
            items = sorted(obj.unpack_var_sequence(tx), key=lambda x: function.call_function(tx, [x], {}).as_python_constant(), reverse=reverse)
        else:
            items = sorted(obj.unpack_var_sequence(tx), key=lambda x: x.as_python_constant(), reverse=reverse)
        return variables.ListVariable(items)