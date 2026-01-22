import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout

            Perform matrix multiplication of C = A * B using CUDA shared memory.

            Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella
            