import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
def test_cat_on_filtered_index():
    df = DataFrame(index=MultiIndex.from_product([[2011, 2012], [1, 2, 3]], names=['year', 'month']))
    df = df.reset_index()
    df = df[df.month > 1]
    str_year = df.year.astype('str')
    str_month = df.month.astype('str')
    str_both = str_year.str.cat(str_month, sep=' ')
    assert str_both.loc[1] == '2011 2'
    str_multiple = str_year.str.cat([str_month, str_month], sep=' ')
    assert str_multiple.loc[1] == '2011 2 2'