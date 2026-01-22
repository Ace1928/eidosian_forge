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
@staticmethod
def call_custom_dict(tx, user_cls, *args, **kwargs):
    if not kwargs:
        if not args:
            args = ({},)
        assert len(args) == 1
        arg = args[0]
        if isinstance(arg, dict):
            return ConstDictVariable(arg, user_cls, mutable_local=MutableLocal())
        elif isinstance(arg, variables.ConstDictVariable):
            return arg.clone(user_cls=user_cls, mutable_local=MutableLocal())
        elif isinstance(arg, (ListVariable, TupleVariable, ListIteratorVariable)):
            items = user_cls()
            for x in arg.unpack_var_sequence(tx):
                k, v = x.unpack_var_sequence(tx)
                k = ConstDictVariable.get_key(k)
                items.update({k: v})
            return ConstDictVariable(items, user_cls, mutable_local=MutableLocal())
    elif not args and kwargs:
        return variables.ConstDictVariable(dict(kwargs), user_cls=user_cls, mutable_local=MutableLocal())
    unimplemented(f'{user_cls.__name__}(): {args} {kwargs}')