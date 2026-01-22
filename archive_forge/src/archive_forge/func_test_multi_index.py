from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_multi_index():
    widget = QgridWidget(df=create_multi_index_df())
    event_history = init_event_history(['filter_dropdown_shown', 'filter_changed', 'sort_changed'], widget=widget)
    widget._handle_qgrid_msg_helper({'type': 'show_filter_dropdown', 'field': 'level_0', 'search_val': None})
    widget._handle_qgrid_msg_helper({'type': 'show_filter_dropdown', 'field': 3, 'search_val': None})
    widget._handle_qgrid_msg_helper({'type': 'change_filter', 'field': 3, 'filter_info': {'field': 3, 'type': 'slider', 'min': -0.111, 'max': None}})
    widget._handle_qgrid_msg_helper({'type': 'change_filter', 'field': 3, 'filter_info': {'field': 3, 'type': 'slider', 'min': None, 'max': None}})
    widget._handle_qgrid_msg_helper({'type': 'change_filter', 'field': 'level_1', 'filter_info': {'field': 'level_1', 'type': 'text', 'selected': [0], 'excluded': []}})
    widget._handle_qgrid_msg_helper({'type': 'change_sort', 'sort_field': 3, 'sort_ascending': True})
    widget._handle_qgrid_msg_helper({'type': 'change_sort', 'sort_field': 'level_0', 'sort_ascending': True})
    assert event_history == [{'name': 'filter_dropdown_shown', 'column': 'level_0'}, {'name': 'filter_dropdown_shown', 'column': 3}, {'name': 'filter_changed', 'column': 3}, {'name': 'filter_changed', 'column': 3}, {'name': 'filter_changed', 'column': 'level_1'}, {'name': 'sort_changed', 'old': {'column': None, 'ascending': True}, 'new': {'column': 3, 'ascending': True}}, {'name': 'sort_changed', 'old': {'column': 3, 'ascending': True}, 'new': {'column': 'level_0', 'ascending': True}}]