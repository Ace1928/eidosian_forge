from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_row_edit_callback():
    sample_df = create_df()

    def can_edit_row(row):
        return row['E'] == 'train' and row['F'] == 'bar'
    view = QgridWidget(df=sample_df, row_edit_callback=can_edit_row)
    view._handle_qgrid_msg_helper({'type': 'change_sort', 'sort_field': 'index', 'sort_ascending': True})
    expected_dict = {0: False, 1: True, 2: False, 3: False}
    assert expected_dict == view._editable_rows