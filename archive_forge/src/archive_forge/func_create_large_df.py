from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def create_large_df(size=10000):
    large_df = pd.DataFrame(np.random.randn(size, 4), columns=list('ABCD'))
    large_df['B (as str)'] = large_df['B'].map(lambda x: str(x))
    return large_df