import numpy as np
from numba.core import config
from numba.cuda.testing import CUDATestCase
from numba import cuda
def make_kernel(vtype):
    """
    Returns a jit compiled kernel that constructs a vector types of
    the given type, using the exact number of primitive types to
    construct the vector type.
    """
    vobj = vtype.user_facing_object
    base_type = vtype.base_type

    def kernel_1elem(res):
        v = vobj(base_type(0))
        res[0] = v.x

    def kernel_2elem(res):
        v = vobj(base_type(0), base_type(1))
        res[0] = v.x
        res[1] = v.y

    def kernel_3elem(res):
        v = vobj(base_type(0), base_type(1), base_type(2))
        res[0] = v.x
        res[1] = v.y
        res[2] = v.z

    def kernel_4elem(res):
        v = vobj(base_type(0), base_type(1), base_type(2), base_type(3))
        res[0] = v.x
        res[1] = v.y
        res[2] = v.z
        res[3] = v.w
    host_function = {1: kernel_1elem, 2: kernel_2elem, 3: kernel_3elem, 4: kernel_4elem}[vtype.num_elements]
    return cuda.jit(host_function)