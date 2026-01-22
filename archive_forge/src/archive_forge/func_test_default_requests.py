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
def test_default_requests():

    class OddEstimator(BaseEstimator):
        __metadata_request__fit = {'sample_weight': True}
    odd_request = get_routing_for_object(OddEstimator())
    assert odd_request.fit.requests == {'sample_weight': True}
    assert not len(get_routing_for_object(NonConsumingClassifier()).fit.requests)
    assert_request_is_empty(NonConsumingClassifier().get_metadata_routing())
    trs_request = get_routing_for_object(ConsumingTransformer())
    assert trs_request.fit.requests == {'sample_weight': None, 'metadata': None}
    assert trs_request.transform.requests == {'metadata': None, 'sample_weight': None}
    assert_request_is_empty(trs_request)
    est_request = get_routing_for_object(ConsumingClassifier())
    assert est_request.fit.requests == {'sample_weight': None, 'metadata': None}
    assert_request_is_empty(est_request)