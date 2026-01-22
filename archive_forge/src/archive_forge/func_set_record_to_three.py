import numpy as np
from numba import cuda
from numba.cuda.args import wrap_arg
from numba.cuda.testing import CUDATestCase
import unittest
def set_record_to_three(rec):
    rec[0]['b'] = 3