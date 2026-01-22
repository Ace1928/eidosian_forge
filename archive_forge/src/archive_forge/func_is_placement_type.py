import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable
@staticmethod
def is_placement_type(value):
    if not DistributedVariable.is_available():
        return False
    from torch.distributed._tensor.placement_types import Placement
    return type(value) is type and issubclass(value, Placement)