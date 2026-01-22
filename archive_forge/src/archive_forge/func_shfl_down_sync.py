from .decorators import jit
import numba
@jit(device=True)
def shfl_down_sync(mask, value, delta):
    """
    Shuffles value across the masked warp and returns the value
    from (laneid + delta). If this is outside the warp, then the
    given value is returned.
    """
    return numba.cuda.shfl_sync_intrinsic(mask, 2, value, delta, 31)[0]