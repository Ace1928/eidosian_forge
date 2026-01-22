from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_remove_row_button():
    widget = QgridWidget(df=create_df())
    event_history = init_event_history(['row_removed', 'selection_changed'], widget=widget)
    selected_rows = [1, 2]
    widget._handle_qgrid_msg_helper({'rows': selected_rows, 'type': 'change_selection'})
    widget._handle_qgrid_msg_helper({'type': 'remove_row'})
    assert event_history == [{'name': 'selection_changed', 'old': [], 'new': selected_rows, 'source': 'gui'}, {'name': 'row_removed', 'indices': selected_rows, 'source': 'gui'}]