import logging
from typing import cast, List
from ...._dynamo.utils import counters
from ... import config, ir
from ...codecache import code_hash, get_path
from ...ir import ComputedBuffer, CUDATemplateBuffer, Pointwise
from ...scheduler import (
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer
from .cutlass_epilogue_gen import CUTLASSEVTOpNotImplementedError
def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
    if self.is_cuda_cpp_template(node1) and isinstance(node2, SchedulerNode):
        return self._can_fuse_epilogue_impl(cast(CUDATemplateBuffer, node1.node), [], node2.node)
    elif self.is_cuda_cpp_fused_template(node1) and isinstance(node2, SchedulerNode):
        fnode1 = cast(FusedSchedulerNode, node1)
        return self._can_fuse_epilogue_impl(fnode1.get_template_node().node, self._unwrap_epilogue_nodes(fnode1), node2.node)
    return False