import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
@require_context
@contextlib.contextmanager
def mapped(*arylist, **kws):
    """A context manager for temporarily mapping a sequence of host ndarrays.
    """
    assert not kws or 'stream' in kws, "Only accept 'stream' as keyword."
    stream = kws.get('stream', 0)
    pmlist = []
    devarylist = []
    for ary in arylist:
        pm = current_context().mempin(ary, driver.host_pointer(ary), driver.host_memory_size(ary), mapped=True)
        pmlist.append(pm)
        devary = devicearray.from_array_like(ary, gpu_data=pm, stream=stream)
        devarylist.append(devary)
    try:
        if len(devarylist) == 1:
            yield devarylist[0]
        else:
            yield devarylist
    finally:
        for pm in pmlist:
            pm.free()