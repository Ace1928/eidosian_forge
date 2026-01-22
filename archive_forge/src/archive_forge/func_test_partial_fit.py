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
def test_partial_fit():
    X = Xdigits.copy()
    rbm = BernoulliRBM(n_components=64, learning_rate=0.1, batch_size=20, random_state=9)
    n_samples = X.shape[0]
    n_batches = int(np.ceil(float(n_samples) / rbm.batch_size))
    batch_slices = np.array_split(X, n_batches)
    for i in range(7):
        for batch in batch_slices:
            rbm.partial_fit(batch)
    assert_almost_equal(rbm.score_samples(X).mean(), -21.0, decimal=0)
    assert_array_equal(X, Xdigits)