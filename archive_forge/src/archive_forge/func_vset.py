import common_z3 as CM_Z3
import ctypes
from .z3 import *
def vset(seq, idfun=None, as_list=True):
    """
    order preserving

    >>> vset([[11,2],1, [10,['9',1]],2, 1, [11,2],[3,3],[10,99],1,[10,['9',1]]],idfun=repr)
    [[11, 2], 1, [10, ['9', 1]], 2, [3, 3], [10, 99]]
    """

    def _uniq_normal(seq):
        d_ = {}
        for s in seq:
            if s not in d_:
                d_[s] = None
                yield s

    def _uniq_idfun(seq, idfun):
        d_ = {}
        for s in seq:
            h_ = idfun(s)
            if h_ not in d_:
                d_[h_] = None
                yield s
    if idfun is None:
        res = _uniq_normal(seq)
    else:
        res = _uniq_idfun(seq, idfun)
    return list(res) if as_list else res