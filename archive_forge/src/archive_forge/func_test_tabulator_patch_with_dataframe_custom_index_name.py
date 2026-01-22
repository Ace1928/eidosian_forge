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
def test_tabulator_patch_with_dataframe_custom_index_name(document, comm):
    df = pd.DataFrame(dict(A=[1, 4, 2]), index=['foo1', 'foo2', 'foo3'])
    df.index.name = 'foo'
    df_patch = pd.DataFrame(dict(A=[10]), index=['foo2'])
    df.index.name = 'foo'
    table = Tabulator(df)
    model = table.get_root(document, comm)
    table.patch(df_patch)
    expected = {'foo': np.array(['foo1', 'foo2', 'foo3']), 'A': np.array([1, 10, 2])}
    for col, values in model.source.data.items():
        expected_array = expected[col]
        np.testing.assert_array_equal(values, expected_array)
        if col != 'foo':
            np.testing.assert_array_equal(table.value[col].values, expected[col])