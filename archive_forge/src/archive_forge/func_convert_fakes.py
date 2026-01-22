from contextlib import contextmanager
import numpy as np
from_record_like = None
def convert_fakes(obj):
    if isinstance(obj, FakeWithinKernelCUDAArray):
        obj = obj._item._ary
    return obj