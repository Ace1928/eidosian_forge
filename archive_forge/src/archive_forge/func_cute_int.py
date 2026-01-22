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
def cute_int(self, int_str: str, var_name: str) -> str:
    res = ''
    if int_str in {'1', '1L'}:
        res = 'cute::Int<1>{}'
    else:
        res = int_str
    return f'{res} /* {var_name} */'