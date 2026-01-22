import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
def custom_assert_frame_equal(left, right, *args, **kwargs):
    obj_type = kwargs.get('obj', 'DataFrame')
    tm.assert_index_equal(left.columns, right.columns, exact=kwargs.get('check_column_type', 'equiv'), check_names=kwargs.get('check_names', True), check_exact=kwargs.get('check_exact', False), check_categorical=kwargs.get('check_categorical', True), obj=f'{obj_type}.columns')
    jsons = (left.dtypes == 'json').index
    for col in jsons:
        custom_assert_series_equal(left[col], right[col], *args, **kwargs)
    left = left.drop(columns=jsons)
    right = right.drop(columns=jsons)
    tm.assert_frame_equal(left, right, *args, **kwargs)