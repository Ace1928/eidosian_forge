import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
import numpy as np

    Test compatibility of CPU and GPU functions
    