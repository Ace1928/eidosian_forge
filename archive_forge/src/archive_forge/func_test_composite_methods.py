import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import (
from sklearn.linear_model import LinearRegression
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.validation import check_is_fitted
def test_composite_methods():

    class SimpleEstimator(BaseEstimator):

        def fit(self, X, y, foo=None, bar=None):
            pass

        def predict(self, X, foo=None, bar=None):
            pass

        def transform(self, X, other_param=None):
            pass
    est = SimpleEstimator()
    assert est.get_metadata_routing().fit_transform.requests == {'bar': None, 'foo': None, 'other_param': None}
    assert est.get_metadata_routing().fit_predict.requests == {'bar': None, 'foo': None}
    est.set_fit_request(foo=True, bar='test')
    with pytest.raises(ValueError, match='Conflicting metadata requests for'):
        est.get_metadata_routing().fit_predict
    est.set_predict_request(bar=True)
    with pytest.raises(ValueError, match='Conflicting metadata requests for'):
        est.get_metadata_routing().fit_predict
    est.set_predict_request(foo=True, bar='test')
    est.get_metadata_routing().fit_predict
    est.set_transform_request(other_param=True)
    assert est.get_metadata_routing().fit_transform.requests == {'bar': 'test', 'foo': True, 'other_param': True}