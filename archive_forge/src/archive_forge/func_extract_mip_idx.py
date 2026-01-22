import abc
from typing import List, Tuple
import numpy as np
from cvxpy.reductions.reduction import Reduction
def extract_mip_idx(variables) -> Tuple[List[int], List[int]]:
    """Coalesces bool, int indices for variables.

       The indexing scheme assumes that the variables will be coalesced into
       a single one-dimensional variable, with each variable being reshaped
       in Fortran order.
    """

    def ravel_multi_index(multi_index, x, vert_offset):
        """Ravel a multi-index and add a vertical offset to it.
        """
        ravel_idx = np.ravel_multi_index(multi_index, max(x.shape, (1,)), order='F')
        return [(vert_offset + idx,) for idx in ravel_idx]
    boolean_idx = []
    integer_idx = []
    vert_offset = 0
    for x in variables:
        if x.boolean_idx:
            multi_index = list(zip(*x.boolean_idx))
            boolean_idx += ravel_multi_index(multi_index, x, vert_offset)
        if x.integer_idx:
            multi_index = list(zip(*x.integer_idx))
            integer_idx += ravel_multi_index(multi_index, x, vert_offset)
        vert_offset += x.size
    return (boolean_idx, integer_idx)