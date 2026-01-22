import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data
def test_invalid_table_type(self):
    with pytest.raises(ValueError, match='must be an array of integers'):
        odds_ratio([[1.0, 3.4], [5.0, 9.9]])