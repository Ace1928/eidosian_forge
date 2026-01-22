from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal
from io import StringIO
from textwrap import dedent
import numpy as np
import numpy.testing as npt
import numpy
from numpy.testing import assert_equal
import pandas
import pytest
from statsmodels.imputation import ros
def test_ros_arrays(self):
    result = ros.impute_ros(self.df[self.rescol], self.df[self.cencol], df=None)
    npt.assert_array_almost_equal(sorted(result), sorted(self.expected_final), decimal=self.decimal)