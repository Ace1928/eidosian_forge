from typing import List
from ..scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .triton import TritonScheduling
def codegen_foreach(self, *args, **kwargs):
    return self._triton_scheduling.codegen_foreach(*args, **kwargs)