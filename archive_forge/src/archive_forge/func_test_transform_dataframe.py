from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
def test_transform_dataframe(self):
    names = self.data.index.names
    transformed_dataframe = self.grouping.transform_dataframe(self.data, lambda x: x.mean(), level=0)
    cols = [names[0]] + list(self.data.columns)
    df = self.data.reset_index()[cols].set_index(names[0])
    grouped = df[self.data.columns].groupby(level=0)
    expected = grouped.apply(lambda x: x.mean())
    np.testing.assert_allclose(transformed_dataframe, expected.values)
    if len(names) > 1:
        transformed_dataframe = self.grouping.transform_dataframe(self.data, lambda x: x.mean(), level=1)
        cols = [names[1]] + list(self.data.columns)
        df = self.data.reset_index()[cols].set_index(names[1])
        grouped = df.groupby(level=0)
        expected = grouped.apply(lambda x: x.mean())[self.data.columns]
        np.testing.assert_allclose(transformed_dataframe, expected.values)