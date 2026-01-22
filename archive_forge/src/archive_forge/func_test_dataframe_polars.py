from packaging.version import Version
from unittest import SkipTest
import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import pytest
from hvplot import hvPlotTabular
from hvplot.tests.util import makeDataFrame
@pytest.mark.skipif(skip_polar, reason='polars not installed')
@pytest.mark.parametrize('cast', (pl.DataFrame, pl.LazyFrame))
@frame_kinds
@y_combinations
def test_dataframe_polars(kind, y, cast):
    df = cast(makeDataFrame())
    assert isinstance(df, cast)
    df.hvplot(y=y, kind=kind)