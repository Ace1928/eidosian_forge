from .decorators import jit
import numba
@jit(device=True)
def ballot_sync(mask, predicate):
    """
    Returns a mask of all threads in the warp whose predicate is true,
    and are within the given mask.
    """
    return numba.cuda.vote_sync_intrinsic(mask, 3, predicate)[0]