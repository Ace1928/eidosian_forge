from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_index_categorical():
    df = pd.DataFrame({'foo': np.random.randn(3), 'future_index': [22, 13, 87]})
    df['future_index'] = df['future_index'].astype('category')
    df = df.set_index('future_index')
    widget = QgridWidget(df=df)
    grid_data = json.loads(widget._df_json)['data']
    assert not isinstance(grid_data[0]['future_index'], dict)
    assert not isinstance(grid_data[1]['future_index'], dict)