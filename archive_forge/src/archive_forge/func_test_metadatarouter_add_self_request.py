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
def test_metadatarouter_add_self_request():
    request = MetadataRequest(owner='nested')
    request.fit.add_request(param='param', alias=True)
    router = MetadataRouter(owner='test').add_self_request(request)
    assert str(router._self_request) == str(request)
    assert router._self_request is not request
    est = ConsumingRegressor().set_fit_request(sample_weight='my_weights')
    router = MetadataRouter(owner='test').add_self_request(obj=est)
    assert str(router._self_request) == str(est.get_metadata_routing())
    assert router._self_request is not est.get_metadata_routing()
    est = WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight='nested_weights'))
    router = MetadataRouter(owner='test').add_self_request(obj=est)
    assert str(router._self_request) == str(est._get_metadata_request())
    assert str(router._self_request) != str(est.get_metadata_routing())
    assert router._self_request is not est._get_metadata_request()