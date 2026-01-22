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
@pytest.mark.xfail(reason='See https://github.com/holoviz/panel/issues/3644')
def test_tabulator_selectable_rows_nonallowed_selection_error(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df, selectable_rows=lambda df: [1])
    model = table.get_root(document, comm)
    assert model.selectable_rows == [1]
    with pytest.raises(ValueError):
        table.selection = [0]