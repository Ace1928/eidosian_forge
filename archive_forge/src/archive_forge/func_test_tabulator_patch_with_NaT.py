import asyncio
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models.widgets.tables import (
from packaging.version import Version
from panel.depends import bind
from panel.io.state import set_curdoc
from panel.models.tabulator import CellClickEvent, TableEditEvent
from panel.tests.util import mpl_available, serve_and_request, wait_until
from panel.util import BOKEH_JS_NAT
from panel.widgets import Button, TextInput
from panel.widgets.tables import DataFrame, Tabulator
def test_tabulator_patch_with_NaT(document, comm):
    df = pd.DataFrame(dict(A=pd.to_datetime(['1980-01-01', np.nan])))
    assert df.loc[1, 'A'] is pd.NaT
    table = Tabulator(df)
    model = table.get_root(document, comm)
    table.patch({'A': [(0, pd.NaT)]})
    expected = {'index': np.array([0, 1]), 'A': np.array([BOKEH_JS_NAT, BOKEH_JS_NAT])}
    for col, values in model.source.data.items():
        expected_array = expected[col]
        np.testing.assert_array_equal(values, expected_array)