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
def test_metadata_routing_get_param_names():
    router = MetadataRouter(owner='test').add_self_request(WeightedMetaRegressor(estimator=ConsumingRegressor()).set_fit_request(sample_weight='self_weights')).add(method_mapping='fit', trs=ConsumingTransformer().set_fit_request(sample_weight='transform_weights'))
    assert str(router) == "{'$self_request': {'fit': {'sample_weight': 'self_weights'}, 'score': {'sample_weight': None}}, 'trs': {'mapping': [{'callee': 'fit', 'caller': 'fit'}], 'router': {'fit': {'sample_weight': 'transform_weights', 'metadata': None}, 'transform': {'sample_weight': None, 'metadata': None}, 'inverse_transform': {'sample_weight': None, 'metadata': None}}}}"
    assert router._get_param_names(method='fit', return_alias=True, ignore_self_request=False) == {'transform_weights', 'metadata', 'self_weights'}
    assert router._get_param_names(method='fit', return_alias=False, ignore_self_request=False) == {'sample_weight', 'metadata', 'transform_weights'}
    assert router._get_param_names(method='fit', return_alias=False, ignore_self_request=True) == {'metadata', 'transform_weights'}
    assert router._get_param_names(method='fit', return_alias=True, ignore_self_request=True) == router._get_param_names(method='fit', return_alias=False, ignore_self_request=True)