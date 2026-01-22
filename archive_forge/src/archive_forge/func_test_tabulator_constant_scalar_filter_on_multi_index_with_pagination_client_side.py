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
def test_tabulator_constant_scalar_filter_on_multi_index_with_pagination_client_side(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df.set_index(['A', 'C']), pagination='remote')
    model = table.get_root(document, comm)
    table.filters = [{'field': 'A', 'sorter': 'number', 'type': '=', 'value': 2}, {'field': 'C', 'type': '=', 'value': 'foo3'}]
    expected = {'index': np.array([0]), 'A': np.array([2]), 'C': np.array(['foo3']), 'B': np.array([0]), 'D': np.array(['2009-01-05T00:00:00.000000000'], dtype='datetime64[ns]').astype(np.int64) / 1000000.0}
    for col, values in model.source.data.items():
        np.testing.assert_array_equal(values, expected[col])