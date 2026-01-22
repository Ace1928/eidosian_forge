from __future__ import annotations
from collections.abc import Callable
from itertools import zip_longest
from numbers import Integral
from typing import Any
import numpy as np
from dask import config
from dask.array.chunk import getitem
from dask.array.core import getter, getter_inline, getter_nofancy
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten, reverse_dict
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse, inline_functions
from dask.utils import ensure_dict
def optimize_slices(dsk):
    """Optimize slices

    1.  Fuse repeated slices, like x[5:][2:6] -> x[7:11]
    2.  Remove full slices, like         x[:] -> x

    See also:
        fuse_slice_dict
    """
    fancy_ind_types = (list, np.ndarray)
    dsk = dsk.copy()
    for k, v in dsk.items():
        if (a_task := _is_getter_task(v)):
            get, a, a_index, a_asarray, a_lock = a_task
            while (b_task := _is_getter_task(a)):
                f2, b, b_index, b_asarray, b_lock = b_task
                if a_lock and a_lock is not b_lock:
                    break
                if (type(a_index) is tuple) != (type(b_index) is tuple):
                    break
                if type(a_index) is tuple:
                    indices = b_index + a_index
                    if len(a_index) != len(b_index) and any((i is None for i in indices)):
                        break
                    if f2 is getter_nofancy and any((isinstance(i, fancy_ind_types) for i in indices)):
                        break
                elif f2 is getter_nofancy and (type(a_index) in fancy_ind_types or type(b_index) in fancy_ind_types):
                    break
                try:
                    c_index = fuse_slice(b_index, a_index)
                    get = getter if f2 is getter_inline else f2
                except NotImplementedError:
                    break
                a, a_index, a_lock = (b, c_index, b_lock)
                a_asarray |= b_asarray
            if get not in GETNOREMOVE and (type(a_index) is slice and (not a_index.start) and (a_index.stop is None) and (a_index.step is None) or (type(a_index) is tuple and all((type(s) is slice and (not s.start) and (s.stop is None) and (s.step is None) for s in a_index)))):
                dsk[k] = a
            elif get is getitem or (a_asarray and (not a_lock)):
                dsk[k] = (get, a, a_index)
            else:
                dsk[k] = (get, a, a_index, a_asarray, a_lock)
    return dsk