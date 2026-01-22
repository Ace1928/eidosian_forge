import re
import sys
import warnings
from io import StringIO
import numpy as np
import pytest
from scipy import linalg
from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.filterwarnings("ignore:The multiplicative update \\('mu'\\) solver cannot update zeros present in the initialization", 'ignore:The default value of `n_components` will change')
def test_parameter_checking():
    A = np.ones((2, 2))
    msg = "Invalid beta_loss parameter: solver 'cd' does not handle beta_loss = 1.0"
    with pytest.raises(ValueError, match=msg):
        NMF(solver='cd', beta_loss=1.0).fit(A)
    msg = 'Negative values in data passed to'
    with pytest.raises(ValueError, match=msg):
        NMF().fit(-A)
    clf = NMF(2, tol=0.1).fit(A)
    with pytest.raises(ValueError, match=msg):
        clf.transform(-A)
    with pytest.raises(ValueError, match=msg):
        nmf._initialize_nmf(-A, 2, 'nndsvd')
    for init in ['nndsvd', 'nndsvda', 'nndsvdar']:
        msg = re.escape("init = '{}' can only be used when n_components <= min(n_samples, n_features)".format(init))
        with pytest.raises(ValueError, match=msg):
            NMF(3, init=init).fit(A)
        with pytest.raises(ValueError, match=msg):
            MiniBatchNMF(3, init=init).fit(A)
        with pytest.raises(ValueError, match=msg):
            nmf._initialize_nmf(A, 3, init)