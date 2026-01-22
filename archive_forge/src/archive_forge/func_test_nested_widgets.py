import numpy as np
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider, RadioButtonGroup, TextInput
from holoviews import Dataset, util
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import Curve, Image
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import ParamMethod, Params
def test_nested_widgets():
    df = makeDataFrame()
    column = RadioButtonGroup(value='A', options=list('ABC'))
    ds = Dataset(df)
    transform = util.transform.df_dim('*').groupby(['D', column]).mean()
    params = list(transform.params.values())
    assert len(params) == 1
    assert params[0] == column.param.value
    df1 = transform.apply(ds, keep_index=True, compute=False)
    df2 = df.groupby(['D', 'A']).mean()
    pd.testing.assert_frame_equal(df1, df2)