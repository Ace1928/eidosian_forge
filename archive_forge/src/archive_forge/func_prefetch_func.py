from collections import namedtuple
import sys
import ctypes
import logging
import threading
import numpy as np
from ..base import _LIB
from ..base import c_str_array, mx_uint, py_str
from ..base import DataIterHandle, NDArrayHandle
from ..base import mx_real_t
from ..base import check_call, build_param_doc as _build_param_doc
from ..ndarray import NDArray
from ..ndarray.sparse import CSRNDArray
from ..ndarray import _ndarray_cls
from ..ndarray import array
from ..ndarray import concat, tile
from .utils import _init_data, _has_instance, _getdata_by_idx, _slice_along_batch_axis
def prefetch_func(self, i):
    """Thread entry"""
    while True:
        self.data_taken[i].wait()
        if not self.started:
            break
        try:
            self.next_batch[i] = self.iters[i].next()
        except StopIteration:
            self.next_batch[i] = None
        self.data_taken[i].clear()
        self.data_ready[i].set()