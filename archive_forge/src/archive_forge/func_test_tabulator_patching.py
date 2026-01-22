from __future__ import annotations
import datetime as dt
from contextlib import contextmanager
import numpy as np
import pandas as pd
import param
import pytest
from bokeh.models.widgets.tables import (
from playwright.sync_api import expect
from panel.depends import bind
from panel.io.state import state
from panel.layout.base import Column
from panel.models.tabulator import _TABULATOR_THEMES_MAPPING
from panel.tests.util import get_ctrl_modifier, serve_component, wait_until
from panel.widgets import Select, Tabulator
def test_tabulator_patching(page, df_mixed):
    widget = Tabulator(df_mixed)
    serve_component(page, widget)
    new_vals = {'str': ['AA', 'BB'], 'int': [100, 101]}
    widget.patch({'str': [(0, new_vals['str'][0]), (1, new_vals['str'][1])], 'int': [(slice(0, 2), new_vals['int'])]}, as_index=False)
    for v in new_vals:
        expect(page.locator(f'text="{v}"')).to_have_count(1)
    assert list(widget.value['str'].iloc[[0, 1]]) == new_vals['str']
    assert list(widget.value['int'].iloc[0:2]) == new_vals['int']
    assert df_mixed.equals(widget.current_view)
    assert df_mixed.equals(widget.value)