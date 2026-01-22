import functools
import logging
from typing import Any, Dict, List, Optional
import torch
from torch._inductor.virtualized import V
from .. import config as inductor_config
from ..codegen.cuda.gemm_template import CUTLASSGemmTemplate
from ..lowering import register_lowering
from ..select_algorithm import (
from ..utils import (
from .mm_common import (
def tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype, *, layout=None):
    out_dtype = torch.promote_types(mat3.get_dtype(), torch.int32) if out_dtype is None else out_dtype
    m, n, k, layout, mat1, mat2, mat3 = mm_args(mat1, mat2, mat3, layout=layout, out_dtype=out_dtype)
    choices: List[Dict[Any, Any]] = []
    for config in int8_mm_configs(m, n, k):
        mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2, mat3), layout=layout, **dict(mm_options(config, k, layout), ACC_TYPE='tl.int32'), suffix_args=1, epilogue_fn=V.ops.mul)
    return autotune_select_algorithm('int_mm', choices, [mat1, mat2, mat3], layout)