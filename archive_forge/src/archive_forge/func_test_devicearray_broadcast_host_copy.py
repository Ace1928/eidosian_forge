import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_broadcast_host_copy(self):
    broadsize = 4
    coreshape = (2, 3)
    coresize = np.prod(coreshape)
    core_c = np.arange(coresize).reshape(coreshape, order='C')
    core_f = np.arange(coresize).reshape(coreshape, order='F')
    for dim in range(len(coreshape)):
        newindex = (slice(None),) * dim + (np.newaxis,)
        broadshape = coreshape[:dim] + (broadsize,) + coreshape[dim:]
        broad_c = np.broadcast_to(core_c[newindex], broadshape)
        broad_f = np.broadcast_to(core_f[newindex], broadshape)
        dbroad_c = cuda.to_device(broad_c)
        dbroad_f = cuda.to_device(broad_f)
        np.testing.assert_array_equal(dbroad_c.copy_to_host(), broad_c)
        np.testing.assert_array_equal(dbroad_f.copy_to_host(), broad_f)
        dbroad_c.copy_to_device(broad_f)
        dbroad_f.copy_to_device(broad_c)
        np.testing.assert_array_equal(dbroad_c.copy_to_host(), broad_f)
        np.testing.assert_array_equal(dbroad_f.copy_to_host(), broad_c)