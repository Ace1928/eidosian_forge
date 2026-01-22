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
def is_live(var: Union[MutableLocalBase, VariableTracker]):
    if isinstance(var, AttributeMutationNew):
        return var in live_new_objects
    if isinstance(var, VariableTracker):
        return is_live(var.mutable_local)
    return True