import collections
import dataclasses
import re
import sys
import types
from typing import Counter, Dict, List, Optional
import torch.nn
from . import utils
from .bytecode_transformation import (
from .exc import unimplemented
from .source import AttrSource, Source
from .utils import is_safe_constant, rot_n_helper
from .variables.base import VariableTracker
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch_function import TensorWithTFOverrideVariable
def make_function_with_closure(self, fn_name: str, code: types.CodeType, push_null: bool, num_on_stack=0):
    freevars = code.co_freevars
    assert freevars
    output = self._output
    if sys.version_info >= (3, 11) and push_null:
        output.append(create_instruction('PUSH_NULL'))
        output.extend(self.rot_n(num_on_stack + 1))
    for var in freevars:
        assert var in self.cell_and_freevars()
        output.append(create_instruction('LOAD_CLOSURE', argval=var))
    output.append(create_instruction('BUILD_TUPLE', arg=len(freevars)))
    output.append(self.create_load_const(code))
    if sys.version_info < (3, 11):
        output.append(self.create_load_const(fn_name))
    output.append(create_instruction('MAKE_FUNCTION', arg=8))
    output.extend(self.rot_n(num_on_stack + 1))
    self.clear_tos()