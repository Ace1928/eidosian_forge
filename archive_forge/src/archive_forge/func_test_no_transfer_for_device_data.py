import numpy as np
from collections import namedtuple
from itertools import product
from numba import vectorize
from numba import cuda, int32, float32, float64
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest
def test_no_transfer_for_device_data(self):
    noise = np.random.randn(1, 3, 64, 64).astype(np.float32)
    noise = cuda.to_device(noise)

    def raising_transfer(*args, **kwargs):
        raise CudaAPIError(999, 'Transfer not allowed')
    old_HtoD = getattr(driver, 'cuMemcpyHtoD', None)
    old_DtoH = getattr(driver, 'cuMemcpyDtoH', None)
    setattr(driver, 'cuMemcpyHtoD', raising_transfer)
    setattr(driver, 'cuMemcpyDtoH', raising_transfer)
    with self.assertRaisesRegex(CudaAPIError, 'Transfer not allowed'):
        noise.copy_to_host()
    with self.assertRaisesRegex(CudaAPIError, 'Transfer not allowed'):
        cuda.to_device([1])
    try:

        @vectorize(['float32(float32)'], target='cuda')
        def func(noise):
            return noise + 1.0
        func(noise)
    finally:
        if old_HtoD is not None:
            setattr(driver, 'cuMemcpyHtoD', old_HtoD)
        else:
            del driver.cuMemcpyHtoD
        if old_DtoH is not None:
            setattr(driver, 'cuMemcpyDtoH', old_DtoH)
        else:
            del driver.cuMemcpyDtoH