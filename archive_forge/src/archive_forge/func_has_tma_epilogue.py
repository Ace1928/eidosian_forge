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
def has_tma_epilogue(op) -> bool:
    assert cutlass_utils.try_import_cutlass()
    import cutlass_library.library as cutlass_lib
    result = False
    if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
        epilogue_schedule_str = str(op.epilogue_schedule).split('.')[-1]
        result = epilogue_schedule_str.lower().startswith('tma')
    return result