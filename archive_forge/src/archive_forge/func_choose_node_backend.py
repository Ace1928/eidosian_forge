from typing import List
from ..scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .triton import TritonScheduling
def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
    if self._cuda_cpp_scheduling.is_cuda_cpp_template(node) or self._cuda_cpp_scheduling.is_cuda_cpp_fused_template(node):
        return self._cuda_cpp_scheduling
    return self._triton_scheduling