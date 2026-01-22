from unittest import TestCase, SkipTest
import holoviews as hv
import pandas as pd
import pytest
from packaging.version import Version
from parameterized import parameterized
from hvplot.converter import HoloViewsConverter
from hvplot.plotting import plot
from hvplot.tests.util import makeDataFrame
def test_pandas_dataframe_plot_does_not_implement_pie(self):
    df = pd.DataFrame({'a': [0, 1, 2], 'b': [5, 7, 2]})
    with self.assertRaisesRegex(NotImplementedError, 'pie'):
        df.plot.pie(y='a')