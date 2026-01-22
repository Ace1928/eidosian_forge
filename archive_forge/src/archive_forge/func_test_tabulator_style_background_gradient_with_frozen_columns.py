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
@mpl_available
def test_tabulator_style_background_gradient_with_frozen_columns(document, comm):
    df = pd.DataFrame(np.random.rand(3, 5), columns=list('ABCDE'))
    table = Tabulator(df, frozen_columns=['A'])
    table.style.background_gradient(cmap='RdYlGn_r', vmin=0, vmax=0.5, subset=['A', 'C', 'D'])
    model = table.get_root(document, comm)
    assert list(model.cell_styles['data'][0]) == [1, 4, 5]