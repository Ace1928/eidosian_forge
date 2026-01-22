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
def test_tabulator_stream_dataframe(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df)
    model = table.get_root(document, comm)
    stream_value = pd.DataFrame({'A': [5, 6], 'B': [1, 0], 'C': ['foo6', 'foo7'], 'D': [dt.datetime(2009, 1, 8), dt.datetime(2009, 1, 9)]})
    table.stream(stream_value)
    assert len(table.value) == 7
    expected = {'index': np.array([0, 1, 2, 3, 4, 5, 6]), 'A': np.array([0, 1, 2, 3, 4, 5, 6]), 'B': np.array([0, 1, 0, 1, 0, 1, 0]), 'C': np.array(['foo1', 'foo2', 'foo3', 'foo4', 'foo5', 'foo6', 'foo7']), 'D': np.array(['2009-01-01T00:00:00.000000000', '2009-01-02T00:00:00.000000000', '2009-01-05T00:00:00.000000000', '2009-01-06T00:00:00.000000000', '2009-01-07T00:00:00.000000000', '2009-01-08T00:00:00.000000000', '2009-01-09T00:00:00.000000000'], dtype='datetime64[ns]').astype(np.int64) / 1000000.0}
    for col, values in model.source.data.items():
        np.testing.assert_array_equal(values, expected[col])