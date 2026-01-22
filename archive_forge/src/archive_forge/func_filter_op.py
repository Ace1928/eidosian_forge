import copy
import logging
import re
from typing import cast, Dict, List, Optional, Tuple
from ...config import cuda as inductor_cuda_config
from ...ir import Buffer, CUDATemplateBuffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer
from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
from .cutlass_epilogue_gen import (
def filter_op(self, op: 'cutlass_library.gemm_op.GemmOperation') -> 'cutlass_library.gemm_op.GemmOperation':
    assert cutlass_utils.try_import_cutlass()
    import cutlass_library.library as cutlass_lib
    if op.tile_description.math_instruction.opcode_class == cutlass_lib.OpcodeClass.Simt:
        return None
    if op.gemm_kind not in {cutlass_lib.GemmKind.Universal, cutlass_lib.GemmKind.Universal3x}:
        return None
    X = self.input_nodes[0]
    W = self.input_nodes[1]
    accumulator_torch_dtype = cutlass_utils.get_accumulator_dtype([X.get_dtype(), W.get_dtype()])
    if not (cutlass_utils.dtype_match(X.get_dtype(), op.A.element) and cutlass_utils.dtype_match(W.get_dtype(), op.B.element) and cutlass_utils.dtype_match(self.output_node.get_layout().dtype, op.C.element) and cutlass_utils.dtype_match(accumulator_torch_dtype, op.accumulator_type())):
        return None
    if not (self.layout_match(X.get_layout(), op.A.layout) and self.layout_match(W.get_layout(), op.B.layout)):
        return None
    op = copy.deepcopy(op)
    op.D.layout = CUTLASSGemmTemplate.cutlass_layout(self.output_node.get_layout())
    if not (self.set_alignment(X.get_layout(), op.A) and self.set_alignment(W.get_layout(), op.B) and self.set_alignment(self.output_node.get_layout(), op.D)):
        return None
    op.element_epilogue = op.accumulator_type()
    if len(self.input_nodes) >= 3 and self.input_nodes[2] is not None:
        Bias = self.input_nodes[2]
        bias_layout = CUTLASSGemmTemplate.cutlass_layout(Bias.get_layout())
        if op.gemm_kind != cutlass_lib.GemmKind.Universal3x:
            if bias_layout != op.D.layout:
                return None
        else:
            op.C.layout = bias_layout
        if not self.set_alignment(Bias.get_layout(), op.C):
            return None
    elif op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
        op.C.element = cutlass_lib.DataType.void
    else:
        op.C.layout = op.D.layout
    supports_evt: bool = self.supports_evt(op)
    if self.can_fuse_epilogue is not None and self.can_fuse_epilogue != supports_evt:
        return None
    if inductor_cuda_config.cutlass_only_evt_capable_ops and (not supports_evt):
        return None
    return op