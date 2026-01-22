from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register_lazy('numba')
def register_numba():
    import numba.cuda

    @sizeof.register(numba.cuda.cudadrv.devicearray.DeviceNDArray)
    def sizeof_numba_devicendarray(x):
        return int(x.nbytes)