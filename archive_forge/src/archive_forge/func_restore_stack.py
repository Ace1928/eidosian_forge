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
def restore_stack(self, stack_values):
    prior = self.mutable_side_effects_from_source
    self.mutable_side_effects_from_source = True
    try:
        self.foreach(stack_values)
    finally:
        self.mutable_side_effects_from_source = prior