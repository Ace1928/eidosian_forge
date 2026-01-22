from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_instance_created():
    event_history = init_event_history(All)
    qgrid_widget = show_grid(create_df())
    assert event_history == [{'name': 'instance_created'}]
    assert qgrid_widget.id