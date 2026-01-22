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
def store_cell(self, cellvar, value):
    assert isinstance(cellvar, variables.NewCellVariable)
    assert isinstance(value, variables.VariableTracker)
    self.store_attr(cellvar, 'cell_contents', value)