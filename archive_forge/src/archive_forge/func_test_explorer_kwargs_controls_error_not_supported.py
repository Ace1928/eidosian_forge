import re
from textwrap import dedent
import holoviews as hv
import pandas as pd
import hvplot.pandas
import hvplot.xarray
import xarray as xr
import pytest
from bokeh.sampledata import penguins
from hvplot.ui import hvDataFrameExplorer, hvGridExplorer
def test_explorer_kwargs_controls_error_not_supported():
    with pytest.raises(TypeError, match=re.escape("__init__() got keyword(s) not supported by any control: {'not_a_control_kwarg': None}")):
        hvplot.explorer(df, title='Dummy title', not_a_control_kwarg=None)