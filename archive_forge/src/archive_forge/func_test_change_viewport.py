from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_change_viewport():
    widget = QgridWidget(df=create_large_df())
    event_history = init_event_history(All)
    widget._handle_qgrid_msg_helper({'type': 'change_viewport', 'top': 7124, 'bottom': 7136})
    assert event_history == [{'name': 'json_updated', 'triggered_by': 'change_viewport', 'range': (7024, 7224)}, {'name': 'viewport_changed', 'old': (0, 100), 'new': (7124, 7136)}]