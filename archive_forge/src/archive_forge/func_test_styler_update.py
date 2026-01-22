import sys
from collections import OrderedDict
import param
from bokeh.models import Div
from panel.depends import bind
from panel.io.notebook import render_mimebundle
from panel.pane import PaneBase
from panel.tests.util import mpl_available
from panel.util import (
@mpl_available
def test_styler_update(dataframe):
    styler = dataframe.style.background_gradient('Reds')
    new_df = dataframe.iloc[:, :2]
    new_style = new_df.style
    new_style._todo = styler_update(styler, new_df)
    new_style._compute()
    assert dict(new_style.ctx) == {(0, 0): [('background-color', '#fff5f0'), ('color', '#000000')], (0, 1): [('background-color', '#fff5f0'), ('color', '#000000')], (1, 0): [('background-color', '#fb694a'), ('color', '#f1f1f1')], (1, 1): [('background-color', '#fb694a'), ('color', '#f1f1f1')], (2, 0): [('background-color', '#67000d'), ('color', '#f1f1f1')], (2, 1): [('background-color', '#67000d'), ('color', '#f1f1f1')]}