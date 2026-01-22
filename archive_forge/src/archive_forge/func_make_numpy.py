import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def make_numpy(name):

    def impl(*args, **kwargs):
        ret_type = args[0].type
        ret_dtype = args[0].dtype
        args = [arg.handle.data for arg in args]
        kwargs = {k: v.handle.data for k, v in kwargs.items()}
        ret = getattr(np, mapping[name])(*args, **kwargs)
        ret = tl.core.tensor(TensorHandle(ret, ret_dtype), ret_type)
        return ret
    return impl