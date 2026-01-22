import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def func_mt(*args):
    length = len(args[0])
    result = np.empty(length, dtype=np.float64)
    args = (result,) + args
    chunklen = (length + numthreads - 1) // numthreads
    chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args] for i in range(numthreads)]
    threads = [threading.Thread(target=inner_func, args=chunk) for chunk in chunks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return result