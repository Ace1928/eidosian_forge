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
def clone_with_transposed_stride(node: IRNode) -> IRNode:
    old_layout = node.get_layout()
    new_stride = list(old_layout.stride)
    new_stride[-2], new_stride[-1] = (new_stride[-1], new_stride[-2])
    new_layout = FixedLayout(old_layout.device, old_layout.dtype, list(old_layout.size), new_stride, old_layout.offset)
    return Buffer(node.get_name(), new_layout)