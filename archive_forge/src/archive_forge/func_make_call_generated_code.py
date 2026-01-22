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
def make_call_generated_code(self, fn_name: str) -> None:
    """Call the generated code function stored in fn_name"""
    self.extend_output(self.load_function_name(fn_name, True))
    graphargs = self.tx.output.graphargs
    for arg in graphargs:
        if arg.is_unspecialized:
            self.extend_output([self.create_load_python_module(torch, True), self.create_load_attr('as_tensor')])
            self.extend_output(arg.load(self))
            self.extend_output(create_call_function(1, False))
        else:
            self.extend_output(arg.load(self))
    self.extend_output(create_call_function(len(graphargs), False))