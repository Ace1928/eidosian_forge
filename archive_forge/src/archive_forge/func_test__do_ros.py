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
def test__do_ros():
    expected = numpy.array([3.11279729, 3.60634338, 4.04602788, 4.04602788, 4.71008116, 6.14010906, 6.97841457, 2.0, 4.2, 4.62, 5.57, 5.66, 5.86, 6.65, 6.78, 6.79, 7.5, 7.5, 7.5, 8.63, 8.71, 8.99, 9.85, 10.82, 11.25, 11.25, 12.2, 14.92, 16.77, 17.81, 19.16, 19.19, 19.64, 20.18, 22.97])
    df = load_basic_data()
    df = ros._do_ros(df, 'conc', 'censored', numpy.log, numpy.exp)
    result = df['final'].values
    npt.assert_array_almost_equal(result, expected)