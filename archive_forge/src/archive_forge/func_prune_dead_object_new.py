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
def prune_dead_object_new(self, tx):
    live_new_objects = set()
    skip_obj = None

    def visit(var: VariableTracker):
        if isinstance(var.mutable_local, AttributeMutationNew) and var.mutable_local is not skip_obj:
            live_new_objects.add(var.mutable_local)
        return var

    def is_live(var: Union[MutableLocalBase, VariableTracker]):
        if isinstance(var, AttributeMutationNew):
            return var in live_new_objects
        if isinstance(var, VariableTracker):
            return is_live(var.mutable_local)
        return True
    VariableTracker.apply(visit, (tx.stack, tx.symbolic_locals))
    for var in self.id_to_variable.values():
        if not isinstance(var.mutable_local, AttributeMutationNew):
            VariableTracker.apply(visit, var)
    for skip_obj, setattrs in self.store_attr_mutations.items():
        VariableTracker.apply(visit, setattrs)
    self.id_to_variable = {k: v for k, v in self.id_to_variable.items() if is_live(v)}
    self.store_attr_mutations = {k: v for k, v in self.store_attr_mutations.items() if is_live(k)}