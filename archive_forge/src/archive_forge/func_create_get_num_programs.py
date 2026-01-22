import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_get_num_programs(self, axis):
    return TensorHandle(np.array([self.grid_dim[axis]], dtype=np.int32), tl.int32)