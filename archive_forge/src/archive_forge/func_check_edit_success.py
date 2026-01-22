from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def check_edit_success(widget, col_name, row_index, old_val_obj, old_val_json, new_val_obj, new_val_json):
    event_history = init_event_history('cell_edited', widget)
    grid_data = json.loads(widget._df_json)['data']
    assert grid_data[row_index][str(col_name)] == old_val_json
    widget._handle_qgrid_msg_helper({'column': col_name, 'row_index': row_index, 'type': 'edit_cell', 'unfiltered_index': row_index, 'value': new_val_json})
    expected_index_val = widget._df.index[row_index]
    assert event_history == [{'name': 'cell_edited', 'index': expected_index_val, 'column': col_name, 'old': old_val_obj, 'new': new_val_obj, 'source': 'gui'}]
    assert widget._df[col_name][row_index] == new_val_obj
    widget._update_table(fire_data_change_event=False)
    grid_data = json.loads(widget._df_json)['data']
    assert grid_data[row_index][str(col_name)] == new_val_json