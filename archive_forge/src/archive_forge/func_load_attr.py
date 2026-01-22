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
def load_attr(self, item, name, deleted_ok=False):
    assert self.is_attribute_mutation(item)
    result = self.store_attr_mutations[item.mutable_local][name]
    if not deleted_ok and isinstance(result, variables.DeletedVariable):
        unimplemented('read deleted attribute')
    return result