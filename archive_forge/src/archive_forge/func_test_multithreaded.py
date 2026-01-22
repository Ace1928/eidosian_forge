from numba import cuda
import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import threading
import unittest
@skip_on_cudasim('Simulator does not support multiple threads')
def test_multithreaded(self):

    def work(gpu, dA, results, ridx):
        try:
            with gpu:
                arr = dA.copy_to_host()
        except Exception as e:
            results[ridx] = e
        else:
            results[ridx] = np.all(arr == np.arange(10))
    dA = cuda.to_device(np.arange(10))
    nthreads = 10
    results = [None] * nthreads
    threads = [threading.Thread(target=work, args=(cuda.gpus.current, dA, results, i)) for i in range(nthreads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for r in results:
        if isinstance(r, BaseException):
            raise r
        else:
            self.assertTrue(r)