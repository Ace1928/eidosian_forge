from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
@_wraps_class_method
def block_rank(self, env, instance):
    """
        block_rank()

        Rank of the calling block within ``[0, num_blocks)``.
        """
    if _runtime.runtimeGetVersion() < 11060:
        raise RuntimeError('block_rank() is supported on CUDA 11.6+')
    _check_include(env, 'cg')
    return _Data(f'{instance.code}.block_rank()', _cuda_types.uint64)