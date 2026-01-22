import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('transformers, remainder, expected_names', [([('bycol1', TransWithNames(), ['d', 'c']), ('bycol2', 'passthrough', ['d'])], 'passthrough', ['bycol1__d', 'bycol1__c', 'bycol2__d', 'remainder__a', 'remainder__b']), ([('bycol1', TransWithNames(), ['d', 'c']), ('bycol2', 'passthrough', ['d'])], 'drop', ['bycol1__d', 'bycol1__c', 'bycol2__d']), ([('bycol1', TransWithNames(), ['b']), ('bycol2', 'drop', ['d'])], 'passthrough', ['bycol1__b', 'remainder__a', 'remainder__c']), ([('bycol1', TransWithNames(['pca1', 'pca2']), ['a', 'b', 'd'])], 'passthrough', ['bycol1__pca1', 'bycol1__pca2', 'remainder__c']), ([('bycol1', TransWithNames(['a', 'b']), ['d']), ('bycol2', 'passthrough', ['b'])], 'drop', ['bycol1__a', 'bycol1__b', 'bycol2__b']), ([('bycol1', TransWithNames([f'pca{i}' for i in range(2)]), ['b']), ('bycol2', TransWithNames([f'pca{i}' for i in range(2)]), ['b'])], 'passthrough', ['bycol1__pca0', 'bycol1__pca1', 'bycol2__pca0', 'bycol2__pca1', 'remainder__a', 'remainder__c', 'remainder__d']), ([('bycol1', 'drop', ['d'])], 'drop', []), ([('bycol1', TransWithNames(), slice(1, 3))], 'drop', ['bycol1__b', 'bycol1__c']), ([('bycol1', TransWithNames(), ['b']), ('bycol2', 'drop', slice(3, 4))], 'passthrough', ['bycol1__b', 'remainder__a', 'remainder__c']), ([('bycol1', TransWithNames(), ['d', 'c']), ('bycol2', 'passthrough', slice(3, 4))], 'passthrough', ['bycol1__d', 'bycol1__c', 'bycol2__d', 'remainder__a', 'remainder__b']), ([('bycol1', TransWithNames(), slice('b', 'c'))], 'drop', ['bycol1__b', 'bycol1__c']), ([('bycol1', TransWithNames(), ['b']), ('bycol2', 'drop', slice('c', 'd'))], 'passthrough', ['bycol1__b', 'remainder__a']), ([('bycol1', TransWithNames(), ['d', 'c']), ('bycol2', 'passthrough', slice('c', 'd'))], 'passthrough', ['bycol1__d', 'bycol1__c', 'bycol2__c', 'bycol2__d', 'remainder__a', 'remainder__b'])])
def test_verbose_feature_names_out_true(transformers, remainder, expected_names):
    """Check feature_names_out for verbose_feature_names_out=True (default)"""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'])
    ct = ColumnTransformer(transformers, remainder=remainder)
    ct.fit(df)
    names = ct.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected_names)