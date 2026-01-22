import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable
def is_constant_pg_functions(value):
    if not DistributedVariable.is_available():
        return False
    from torch.distributed.distributed_c10d import _get_group_tag, get_process_group_ranks
    constant_processgroup_functions = [get_process_group_ranks, _get_group_tag]
    return inspect.isfunction(value) and value in constant_processgroup_functions