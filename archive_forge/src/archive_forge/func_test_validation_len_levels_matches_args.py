import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.stats.contingency import crosstab
def test_validation_len_levels_matches_args():
    with pytest.raises(ValueError, match='number of input sequences'):
        crosstab([0, 1, 1], [8, 8, 9], levels=([0, 1, 2, 3],))