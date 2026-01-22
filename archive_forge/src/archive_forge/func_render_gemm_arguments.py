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
def render_gemm_arguments(self, argument_template: str, epilogue_template: str, should_swap_xw: bool, X: IRNode, W: IRNode, Bias: IRNode, Y: IRNode, alpha: float, beta: float, kernel: CUDATemplateKernel, epilogue_args) -> str:
    options = dict(alpha=self.alpha, beta=self.beta, X=X, W=W, Y=Y, Bias=Bias, template=self, kernel=kernel, M='M', N='N', epilogue_args=epilogue_args)
    if epilogue_template is not None:
        if should_swap_xw:

            def clone_with_transposed_stride(node: IRNode) -> IRNode:
                old_layout = node.get_layout()
                new_stride = list(old_layout.stride)
                new_stride[-2], new_stride[-1] = (new_stride[-1], new_stride[-2])
                new_layout = FixedLayout(old_layout.device, old_layout.dtype, list(old_layout.size), new_stride, old_layout.offset)
                return Buffer(node.get_name(), new_layout)
            new_X = clone_with_transposed_stride(X)
            new_W = clone_with_transposed_stride(W)
            new_Bias = clone_with_transposed_stride(Bias)
            new_Y = clone_with_transposed_stride(Y)
            options['X'], options['W'], options['Bias'], options['Y'] = (new_W, new_X, new_Bias, new_Y)
            options['M'], options['N'] = ('N', 'M')
        epilogue_arguments = self._template_from_string(epilogue_template).render(**options)
        arguments = self._template_from_string(argument_template).render(epilogue_arguments=epilogue_arguments, **options)
    else:
        arguments = self._template_from_string(GEMM_ARGS_CUTLASS_2X).render(split_k=1, **options)
    return arguments