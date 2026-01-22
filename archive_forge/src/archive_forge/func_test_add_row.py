from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_add_row():
    event_history = init_event_history(All)
    df = pd.DataFrame({'foo': ['hello'], 'bar': ['world'], 'baz': [42], 'boo': [57]})
    df.set_index('baz', inplace=True, drop=True)
    q = QgridWidget(df=df)
    new_row = [('baz', 43), ('bar', 'new bar'), ('boo', 58), ('foo', 'new foo')]
    q.add_row(new_row)
    assert q._df.loc[43, 'foo'] == 'new foo'
    assert q._df.loc[42, 'foo'] == 'hello'
    assert event_history == [{'name': 'instance_created'}, {'name': 'json_updated', 'range': (0, 100), 'triggered_by': 'add_row'}, {'name': 'row_added', 'index': 43, 'source': 'api'}]