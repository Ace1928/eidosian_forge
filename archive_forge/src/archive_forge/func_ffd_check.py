import numpy as np
import numba
@numba.njit
def ffd_check(a: np.ndarray, c: int, n: int):
    a = np.sort(a)[::-1]
    bins = np.full((n,), c, dtype=a.dtype)
    for size in a:
        not_found = True
        for idx in range(n):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break
        if not_found:
            return False
    return True