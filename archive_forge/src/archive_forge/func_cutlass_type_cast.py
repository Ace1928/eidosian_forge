import functools
import itertools
import logging
from typing import List, Optional
from unittest.mock import patch
import sympy
import torch
from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel
def cutlass_type_cast(self, node: IRNode, ptr: str) -> str:
    if node is None:
        return ptr
    else:
        return f'({self._DTYPE_TO_CUTLASS.get(node.get_dtype())}*)({ptr})'