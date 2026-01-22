import sys
from io import StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.linalg import block_diag
from scipy.special import psi
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._online_lda_fast import (
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def check_verbosity(verbose, evaluate_every, expected_lines, expected_perplexities, csr_container):
    n_components, X = _build_sparse_array(csr_container)
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=3, learning_method='batch', verbose=verbose, evaluate_every=evaluate_every, random_state=0)
    out = StringIO()
    old_out, sys.stdout = (sys.stdout, out)
    try:
        lda.fit(X)
    finally:
        sys.stdout = old_out
    n_lines = out.getvalue().count('\n')
    n_perplexity = out.getvalue().count('perplexity')
    assert expected_lines == n_lines
    assert expected_perplexities == n_perplexity