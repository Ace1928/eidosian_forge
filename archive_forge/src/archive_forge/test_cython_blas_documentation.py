import numpy as np
from numpy.testing import (assert_allclose,
import scipy.linalg.cython_blas as blas
 Test the function pointers that are expected to fail on
    Mac OS X without the additional entry statement in their definitions
    in fblas_l1.pyf.src. 