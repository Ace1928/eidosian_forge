import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def patch_attr(obj, name, member, builder):
    new_member = lambda *args, member=member, **kwargs: member(*args, **{k: v for k, v in kwargs.items() if k != '_builder'}, _builder=builder)
    setattr(obj, name, new_member)