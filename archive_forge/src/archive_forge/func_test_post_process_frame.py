import textwrap
from io import BytesIO
import pytest
from sklearn.datasets._arff_parser import (
@pytest.mark.parametrize('feature_names, target_names', [(['col_int_as_integer', 'col_int_as_numeric', 'col_float_as_real', 'col_float_as_numeric'], ['col_categorical', 'col_string']), (['col_int_as_integer', 'col_int_as_numeric', 'col_float_as_real', 'col_float_as_numeric'], ['col_categorical']), (['col_int_as_integer', 'col_int_as_numeric', 'col_float_as_real', 'col_float_as_numeric'], [])])
def test_post_process_frame(feature_names, target_names):
    """Check the behaviour of the post-processing function for splitting a dataframe."""
    pd = pytest.importorskip('pandas')
    X_original = pd.DataFrame({'col_int_as_integer': [1, 2, 3], 'col_int_as_numeric': [1, 2, 3], 'col_float_as_real': [1.0, 2.0, 3.0], 'col_float_as_numeric': [1.0, 2.0, 3.0], 'col_categorical': ['a', 'b', 'c'], 'col_string': ['a', 'b', 'c']})
    X, y = _post_process_frame(X_original, feature_names, target_names)
    assert isinstance(X, pd.DataFrame)
    if len(target_names) >= 2:
        assert isinstance(y, pd.DataFrame)
    elif len(target_names) == 1:
        assert isinstance(y, pd.Series)
    else:
        assert y is None