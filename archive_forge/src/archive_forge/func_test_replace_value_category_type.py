from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_value_category_type(self):
    """
        Test for #23305: to ensure category dtypes are maintained
        after replace with direct values
        """
    input_dict = {'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd'], 'col3': [1.5, 2.5, 3.5, 4.5], 'col4': ['cat1', 'cat2', 'cat3', 'cat4'], 'col5': ['obj1', 'obj2', 'obj3', 'obj4']}
    input_df = DataFrame(data=input_dict).astype({'col2': 'category', 'col4': 'category'})
    input_df['col2'] = input_df['col2'].cat.reorder_categories(['a', 'b', 'c', 'd'], ordered=True)
    input_df['col4'] = input_df['col4'].cat.reorder_categories(['cat1', 'cat2', 'cat3', 'cat4'], ordered=True)
    expected_dict = {'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'z'], 'col3': [1.5, 2.5, 3.5, 4.5], 'col4': ['cat1', 'catX', 'cat3', 'cat4'], 'col5': ['obj9', 'obj2', 'obj3', 'obj4']}
    expected = DataFrame(data=expected_dict).astype({'col2': 'category', 'col4': 'category'})
    expected['col2'] = expected['col2'].cat.reorder_categories(['a', 'b', 'c', 'z'], ordered=True)
    expected['col4'] = expected['col4'].cat.reorder_categories(['cat1', 'catX', 'cat3', 'cat4'], ordered=True)
    msg = 'The behavior of Series\\.replace \\(and DataFrame.replace\\) with CategoricalDtype'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        input_df = input_df.replace('d', 'z')
        input_df = input_df.replace('obj1', 'obj9')
        result = input_df.replace('cat2', 'catX')
    tm.assert_frame_equal(result, expected)