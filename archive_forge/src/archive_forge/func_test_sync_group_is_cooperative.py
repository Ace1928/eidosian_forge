from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_unless_cc_60
@skip_on_cudasim("Simulator doesn't differentiate between normal and cooperative kernels")
def test_sync_group_is_cooperative(self):
    A = np.full(1, fill_value=np.nan)
    sync_group[1, 1](A)
    for key, overload in sync_group.overloads.items():
        self.assertTrue(overload.cooperative)