import re
import sys
from io import StringIO
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_sparse_and_verbose(csc_container):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    X = csc_container([[0.0], [1.0]])
    rbm = BernoulliRBM(n_components=2, batch_size=2, n_iter=1, random_state=42, verbose=True)
    try:
        rbm.fit(X)
        s = sys.stdout.getvalue()
        assert re.match('\\[BernoulliRBM\\] Iteration 1, pseudo-likelihood = -?(\\d)+(\\.\\d+)?, time = (\\d|\\.)+s', s)
    finally:
        sys.stdout = old_stdout