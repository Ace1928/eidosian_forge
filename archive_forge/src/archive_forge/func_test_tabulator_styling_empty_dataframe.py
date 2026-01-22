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
def test_tabulator_styling_empty_dataframe(document, comm):
    df = pd.DataFrame(columns=['A', 'B', 'C']).astype({'A': float, 'B': str, 'C': int})
    table = Tabulator(df)
    table.style.apply(lambda x: ['border-color: #dc3545; border-style: solid' for name, value in x.items()], axis=1)
    model = table.get_root(document, comm)
    assert model.styles == {}
    table.value = pd.DataFrame({'A': [3.14], 'B': ['foo'], 'C': [3]})
    assert model.cell_styles['data'] == {0: {2: [('border-color', '#dc3545'), ('border-style', 'solid')], 3: [('border-color', '#dc3545'), ('border-style', 'solid')], 4: [('border-color', '#dc3545'), ('border-style', 'solid')]}}