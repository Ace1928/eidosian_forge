from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
def test_count_categories(self):
    self.grouping.count_categories(level=0)
    np.testing.assert_equal(self.grouping.counts, self.expected_counts)