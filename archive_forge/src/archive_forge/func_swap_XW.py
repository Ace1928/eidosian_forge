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
@staticmethod
def swap_XW(op: 'cutlass_library.gemm_op.GemmOperation') -> 'cutlass_library.gemm_op.GemmOperation':
    new_op = copy.deepcopy(op)
    new_op.A.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.A.layout)
    new_op.B.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.B.layout)
    new_op.A, new_op.B = (new_op.B, new_op.A)
    new_op.C.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.C.layout)
    new_op.D.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.D.layout)
    return new_op