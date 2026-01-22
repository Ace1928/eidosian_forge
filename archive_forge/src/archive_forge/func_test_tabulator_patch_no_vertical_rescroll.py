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
@pytest.mark.xfail(reason='See https://github.com/holoviz/panel/issues/3249')
def test_tabulator_patch_no_vertical_rescroll(page):
    size = 10
    arr = np.random.choice(list('abcd'), size=size)
    target, new_val = ('X', 'Y')
    arr[-1] = target
    df = pd.DataFrame({'col': arr})
    height, width = (100, 200)
    widget = Tabulator(df, height=height, width=width)
    serve_component(page, widget)
    target_cell = page.locator(f'text="{target}"')
    target_cell.scroll_into_view_if_needed()
    page.wait_for_timeout(400)
    page.mouse.move(x=int(width / 2), y=int(height / 2))
    page.mouse.wheel(delta_x=0, delta_y=10000)
    page.wait_for_timeout(400)
    bb = page.locator(f'text="{target}"').bounding_box()
    widget.patch({'col': [(size - 1, new_val)]})
    page.wait_for_timeout(400)
    assert bb == page.locator(f'text="{new_val}"').bounding_box()