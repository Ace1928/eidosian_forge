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
def codegen_template(self, template_node: BaseSchedulerNode, epilogue_nodes: List[SchedulerNode]):
    """
        Codegen a CUDA template, possibly with fused epilogues
        """
    counters['inductor']['cuda_epilogue_fusion_counter'] += len(epilogue_nodes)
    assert self.is_cuda_cpp_template(template_node), 'Template node passed to CUDAScheduler.codegen_template must be a SchedulerNode that wraps a CUDATemplateBuffer'
    template_node = cast(SchedulerNode, template_node)
    _, (numel, rnumel) = template_node.group
    assert rnumel == 1
    ctb: CUDATemplateBuffer = cast(CUDATemplateBuffer, template_node.node)
    epilogue_ir_nodes: List[ir.Buffer] = [n.node for n in epilogue_nodes]
    assert all((isinstance(n, ir.ComputedBuffer) for n in epilogue_ir_nodes)), 'Epilogue nodes must all be instances of ir.ComputedBuffer'
    kernel, render = ctb.make_kernel_render(ctb, epilogue_nodes=epilogue_ir_nodes)
    with kernel:
        for node in [template_node, *epilogue_nodes]:
            node.mark_run()
        src_code = render()
    with V.set_kernel_handler(kernel):
        node_schedule = [template_node, *epilogue_nodes]
        kernel_name = self.define_kernel(src_code, node_schedule)
    kernel.call_kernel(kernel_name, ctb, epilogue_ir_nodes)
    V.graph.removed_buffers |= kernel.removed_buffers
    self.scheduler.free_buffers()