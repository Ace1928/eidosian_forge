from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
def compute_ufunc_cuda_dtype_body(g: NativeFunctionsGroup, dtype: ScalarType, inner_loops: Dict[UfuncKey, UfunctorSignature], parent_ctx: Sequence[Binding]) -> str:
    body = 'using opmath_t = at::opmath_type<scalar_t>;'
    body += 'if (false) {}\n'
    for config in BinaryScalarSpecializationConfigs:
        if config.ufunc_key not in inner_loops:
            continue
        ufunctor_sig = inner_loops[config.ufunc_key]
        scalar_idx = config.scalar_idx + 1
        ctx: List[Union[Expr, Binding]] = list(parent_ctx)
        ctx.append(Expr(expr=f'iter.scalar_value<opmath_t>({scalar_idx})', type=NamedCType(config.ctor_tensor, BaseCType(opmath_t))))
        ufunctor_ctor_exprs_str = ', '.join((a.expr for a in translate(ctx, ufunctor_sig.arguments().ctor)))
        body += f'else if (iter.is_cpu_scalar({scalar_idx})) {{\n  {ufunctor_sig.name}<scalar_t> ufunctor({ufunctor_ctor_exprs_str});\n  iter.remove_operand({scalar_idx});\n  gpu_kernel(iter, ufunctor);\n}}'
    ufunctor_sig = inner_loops[UfuncKey.CUDAFunctor]
    ufunctor_ctor_exprs_str = ', '.join((a.expr for a in translate(parent_ctx, ufunctor_sig.arguments().ctor)))
    body += f'\nelse {{\n  gpu_kernel(iter, {ufunctor_sig.name}<scalar_t>({ufunctor_ctor_exprs_str}));\n}}\n    '
    return body