import inspect
from typing import Any, Dict, List, Optional, Union
import torch.nn
from . import utils, variables
from .bytecode_transformation import (
from .codegen import PyCodegen
from .exc import unimplemented
from .source import LocalSource, Source
from .utils import nn_module_new, object_new
from .variables.base import (
def store_global(self, gvar: VariableTracker, name: str, value: VariableTracker):
    assert isinstance(gvar, variables.VariableTracker)
    assert isinstance(value, variables.VariableTracker)
    self.store_attr(gvar, name, value)