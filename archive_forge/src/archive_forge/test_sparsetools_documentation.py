import sys
import os
import gc
import threading
import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from scipy.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
from scipy.sparse._sputils import supported_dtypes
from scipy._lib._testutils import check_free_memory
import pytest
from pytest import raises as assert_raises

    Some of the sparsetools routines use dense 2D matrices whose
    total size is not bounded by the nnz of the sparse matrix. These
    routines used to suffer from int32 wraparounds; here, we try to
    check that the wraparounds don't occur any more.
    