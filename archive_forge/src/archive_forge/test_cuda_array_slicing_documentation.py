from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch

    Most of the slicing logic is tested in the cases above, so these
    tests focus on the setting logic.
    