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
@pd_old
def test_tabulator_styling(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df)

    def high_red(value):
        return 'color: red' if value > 2 else 'color: black'
    table.style.applymap(high_red, subset=['A'])
    model = table.get_root(document, comm)
    assert model.cell_styles['data'] == {0: {2: [('color', 'black')]}, 1: {2: [('color', 'black')]}, 2: {2: [('color', 'black')]}, 3: {2: [('color', 'red')]}, 4: {2: [('color', 'red')]}}