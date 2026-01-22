from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def create_interval_index_df():
    td = np.cumsum(np.random.randint(1, 15 * 60, 1000))
    start = pd.Timestamp('2017-04-17')
    df = pd.DataFrame([start + pd.Timedelta(seconds=d) for d in td], columns=['time'], dtype='M8[ns]')
    df['time_bin'] = np.cumsum(np.random.randint(1, 15 * 60, 1000))
    return df