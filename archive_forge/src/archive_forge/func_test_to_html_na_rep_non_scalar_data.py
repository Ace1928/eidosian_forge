from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_to_html_na_rep_non_scalar_data(datapath):
    df = DataFrame([{'a': 1, 'b': [1, 2, 3], 'c': np.nan}])
    result = df.style.format(na_rep='-').to_html(table_uuid='test')
    expected = '<style type="text/css">\n</style>\n<table id="T_test">\n  <thead>\n    <tr>\n      <th class="blank level0" >&nbsp;</th>\n      <th id="T_test_level0_col0" class="col_heading level0 col0" >a</th>\n      <th id="T_test_level0_col1" class="col_heading level0 col1" >b</th>\n      <th id="T_test_level0_col2" class="col_heading level0 col2" >c</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th id="T_test_level0_row0" class="row_heading level0 row0" >0</th>\n      <td id="T_test_row0_col0" class="data row0 col0" >1</td>\n      <td id="T_test_row0_col1" class="data row0 col1" >[1, 2, 3]</td>\n      <td id="T_test_row0_col2" class="data row0 col2" >-</td>\n    </tr>\n  </tbody>\n</table>\n'
    assert result == expected