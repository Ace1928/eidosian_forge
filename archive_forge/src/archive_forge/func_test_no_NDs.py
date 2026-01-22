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
def test_no_NDs(self):
    _df = self.df.copy()
    _df['qual'] = False
    result = ros.cohn_numbers(_df, observations='conc', censorship='qual')
    assert result.shape == (0, 6)