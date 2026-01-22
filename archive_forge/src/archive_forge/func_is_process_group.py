import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable
@staticmethod
def is_process_group(value):
    if not DistributedVariable.is_available():
        return False
    from torch._C._distributed_c10d import ProcessGroup
    from torch.testing._internal.distributed.fake_pg import FakeProcessGroup
    return istype(value, (ProcessGroup, FakeProcessGroup))