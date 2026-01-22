from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_edit_multi_index_df():
    df_multi = create_multi_index_df()
    df_multi.index.set_names('first', level=0, inplace=True)
    view = QgridWidget(df=df_multi)
    old_val = df_multi.loc[('bar', 'two'), 1]
    check_edit_success(view, 1, 1, old_val, round(old_val, pd.get_option('display.precision') - 1), 3.45678, 3.45678)