from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('as_index', [False, True])
@pytest.mark.parametrize('observed', [False, True])
@pytest.mark.parametrize('normalize, name, expected_data', [(False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)), (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))])
def test_categorical_single_grouper_with_only_observed_categories(education_df, as_index, observed, normalize, name, expected_data, request):
    if Version(np.__version__) >= Version('1.25'):
        request.node.add_marker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
    gp = education_df.astype('category').groupby('country', as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)
    expected_index = MultiIndex.from_tuples([('FR', 'male', 'low'), ('FR', 'female', 'high'), ('FR', 'male', 'medium'), ('FR', 'female', 'low'), ('FR', 'female', 'medium'), ('FR', 'male', 'high'), ('US', 'female', 'high'), ('US', 'male', 'low'), ('US', 'female', 'low'), ('US', 'female', 'medium'), ('US', 'male', 'high'), ('US', 'male', 'medium')], names=['country', 'gender', 'education'])
    expected_series = Series(data=expected_data, index=expected_index, name=name)
    for i in range(3):
        expected_series.index = expected_series.index.set_levels(CategoricalIndex(expected_series.index.levels[i]), level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name='proportion' if normalize else 'count')
        tm.assert_frame_equal(result, expected)