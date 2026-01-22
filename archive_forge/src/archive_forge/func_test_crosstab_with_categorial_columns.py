import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_with_categorial_columns(self):
    df = DataFrame({'MAKE': ['Honda', 'Acura', 'Tesla', 'Honda', 'Honda', 'Acura'], 'MODEL': ['Sedan', 'Sedan', 'Electric', 'Pickup', 'Sedan', 'Sedan']})
    categories = ['Sedan', 'Electric', 'Pickup']
    df['MODEL'] = df['MODEL'].astype('category').cat.set_categories(categories)
    result = crosstab(df['MAKE'], df['MODEL'])
    expected_index = Index(['Acura', 'Honda', 'Tesla'], name='MAKE')
    expected_columns = CategoricalIndex(categories, categories=categories, ordered=False, name='MODEL')
    expected_data = [[2, 0, 0], [2, 0, 1], [0, 1, 0]]
    expected = DataFrame(expected_data, index=expected_index, columns=expected_columns)
    tm.assert_frame_equal(result, expected)