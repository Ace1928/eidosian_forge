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
def define_gemm_instance(self, op: 'cutlass_library.gemm_op.GemmOperation', output_buffer_name: str, epilogue_nodes: Optional[List[IRNode]]=None) -> Tuple[str, str]:
    assert cutlass_utils.try_import_cutlass()
    import cutlass_library.gemm_operation as cutlass_gemm_op
    import cutlass_library.library as cutlass_lib
    from torch._inductor.codegen.cuda.cutlass_lib_extensions.gemm_operation_extensions import EmitGemmUniversal3xInstanceWithEVT
    if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            emitter = EmitGemmUniversal3xInstanceWithEVT()
            op.epilogue_functor = lambda epilogue_functor_type_name: self.render_evt_epilogue_declaration(output_buffer_name, epilogue_functor_type_name, epilogue_nodes)
        else:
            emitter = cutlass_gemm_op.EmitGemmUniversal3xInstance()
        op_def = emitter.emit(op)
        pattern = re.compile('\\s*struct\\s(.*?)\\s:')
        decl = [line for line in op_def.split('\n') if 'struct ' in line][-1]
    else:
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            raise RuntimeError('EVT epilogue fusion is not supported for Cutlass 2.x ops.')
        emitter = cutlass_gemm_op.EmitGemmInstance()
        op_def = emitter.emit(op)
        op_def = op_def.replace('cutlass::gemm::device::Gemm', 'cutlass::gemm::device::GemmUniversal')
        op_def = op_def.replace('false,', '')
        pattern = re.compile('\\s*using\\s(.*?)\\s=')
        decl = op_def.split('\n')[2]
    match = pattern.match(decl)
    if match is None:
        raise RuntimeError('Invalid Gemm config: \n' + op_def)
    op_type = match.groups()[0]
    if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
        op_def += f'\n  using {op_type}_device_type = cutlass::gemm::device::GemmUniversalAdapter<{op_type}>;\n'
        op_type = f'{op_type}_device_type'
    return (op_def, op_type)