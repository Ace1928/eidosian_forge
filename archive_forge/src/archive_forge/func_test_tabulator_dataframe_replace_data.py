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
def test_tabulator_dataframe_replace_data(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df)
    model = table.get_root(document, comm)
    custom_df = pd.DataFrame({'C_l0_g0': {'R_l0_g0': 'R0C0', 'R_l0_g1': 'R1C0'}, 'C_l0_g1': {'R_l0_g0': 'R0C1', 'R_l0_g1': 'R1C1'}})
    custom_df.index.name = 'R0'
    custom_df.columns.name = 'C0'
    table.value = custom_df
    assert len(model.columns) == 3
    c1, c2, c3 = model.columns
    assert c1.field == 'R0'
    assert c2.field == 'C_l0_g0'
    assert c3.field == 'C_l0_g1'
    assert model.configuration == {'columns': [{'field': 'R0'}, {'field': 'C_l0_g0'}, {'field': 'C_l0_g1'}], 'selectable': True, 'dataTree': False}
    expected = {'C_l0_g0': np.array(['R0C0', 'R1C0'], dtype=object), 'C_l0_g1': np.array(['R0C1', 'R1C1'], dtype=object), 'R0': np.array(['R_l0_g0', 'R_l0_g1'], dtype=object)}
    for col, values in model.source.data.items():
        np.testing.assert_array_equal(values, expected[col])