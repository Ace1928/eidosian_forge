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
def test_uncensored_2(self):
    row = {'censored': False, 'det_limit_index': 2, 'rank': 12}
    result = ros._ros_plot_pos(row, 'censored', self.cohn)
    assert_equal(result, 0.5189931350114417)