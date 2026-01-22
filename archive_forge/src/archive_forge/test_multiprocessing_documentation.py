import os
import multiprocessing as mp
import numpy as np
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest

        Test fork detection.
        