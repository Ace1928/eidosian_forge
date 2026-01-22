import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('Transformer', [MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer, Normalizer, Binarizer])
def test_one_to_one_features_pandas(Transformer):
    """Check one-to-one transformers give correct feature names."""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    tr = Transformer().fit(df)
    names_out_df_default = tr.get_feature_names_out()
    assert_array_equal(names_out_df_default, iris.feature_names)
    names_out_df_valid_in = tr.get_feature_names_out(iris.feature_names)
    assert_array_equal(names_out_df_valid_in, iris.feature_names)
    msg = re.escape('input_features is not equal to feature_names_in_')
    with pytest.raises(ValueError, match=msg):
        invalid_names = list('abcd')
        tr.get_feature_names_out(invalid_names)