from .decorators import jit
import numba
@jit(device=True)
def shfl_xor_sync(mask, value, lane_mask):
    """
    Shuffles value across the masked warp and returns the value
    from (laneid ^ lane_mask).
    """
    return numba.cuda.shfl_sync_intrinsic(mask, 3, value, lane_mask, 31)[0]