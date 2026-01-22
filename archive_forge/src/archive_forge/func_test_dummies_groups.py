from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
@pytest.mark.smoke
def test_dummies_groups(self):
    self.grouping.dummies_groups()
    if len(self.grouping.group_names) > 1:
        self.grouping.dummies_groups(level=1)