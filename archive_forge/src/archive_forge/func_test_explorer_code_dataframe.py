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
def test_explorer_code_dataframe():
    explorer = hvplot.explorer(df, x='bill_length_mm', kind='points')
    assert explorer.code == dedent("        df.hvplot(\n            kind='points',\n            x='bill_length_mm',\n            y='species',\n            legend='bottom_right',\n            widget_location='bottom',\n        )")
    assert explorer._code_pane.object == dedent("        ```python\n        df.hvplot(\n            kind='points',\n            x='bill_length_mm',\n            y='species',\n            legend='bottom_right',\n            widget_location='bottom',\n        )\n        ```")