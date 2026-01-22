from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('as_index', [True, False])
@pytest.mark.parametrize('observed, expected_index', [(False, [('FR', 'high', 'female'), ('FR', 'high', 'male'), ('FR', 'low', 'male'), ('FR', 'low', 'female'), ('FR', 'medium', 'male'), ('FR', 'medium', 'female'), ('US', 'high', 'female'), ('US', 'high', 'male'), ('US', 'low', 'male'), ('US', 'low', 'female'), ('US', 'medium', 'female'), ('US', 'medium', 'male')]), (True, [('FR', 'high', 'female'), ('FR', 'low', 'male'), ('FR', 'medium', 'male'), ('US', 'high', 'female'), ('US', 'low', 'male')])])
@pytest.mark.parametrize('normalize, name, expected_data', [(False, 'count', np.array([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64)), (True, 'proportion', np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]))])
def test_categorical_multiple_groupers(education_df, as_index, observed, expected_index, normalize, name, expected_data):
    education_df = education_df.copy()
    education_df['country'] = education_df['country'].astype('category')
    education_df['education'] = education_df['education'].astype('category')
    gp = education_df.groupby(['country', 'education'], as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)
    expected_series = Series(data=expected_data[expected_data > 0.0] if observed else expected_data, index=MultiIndex.from_tuples(expected_index, names=['country', 'education', 'gender']), name=name)
    for i in range(2):
        expected_series.index = expected_series.index.set_levels(CategoricalIndex(expected_series.index.levels[i]), level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name='proportion' if normalize else 'count')
        tm.assert_frame_equal(result, expected)