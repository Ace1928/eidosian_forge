import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
@pytest.mark.parametrize('constructor_name', ['sparse_csr', 'sparse_csc', pytest.param('sparse_csr_array', marks=pytest.mark.skipif(sp_version < parse_version('1.8'), reason='sparse arrays are available as of scipy 1.8.0')), pytest.param('sparse_csc_array', marks=pytest.mark.skipif(sp_version < parse_version('1.8'), reason='sparse arrays are available as of scipy 1.8.0'))])
def test_convert_container_sparse_to_sparse(constructor_name):
    """Non-regression test to check that we can still convert a sparse container
    from a given format to another format.
    """
    X_sparse = sparse.random(10, 10, density=0.1, format='csr')
    _convert_container(X_sparse, constructor_name)