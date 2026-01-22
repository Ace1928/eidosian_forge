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
def call_type(self, tx, obj: VariableTracker):
    from .builder import SourcelessBuilder, VariableBuilder
    try:
        py_type = obj.python_type()
    except NotImplementedError as error:
        raise UserError(UserErrorType.INVALID_INPUT, str(error), case_name='unknown_python_type') from None
    if obj.source is None:
        return SourcelessBuilder()(tx, py_type)
    else:
        return VariableBuilder(tx, TypeSource(obj.source))(py_type)