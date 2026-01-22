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
def has_pending_mutation(self, item):
    return self.is_attribute_mutation(item) and bool(self.store_attr_mutations.get(item.mutable_local))