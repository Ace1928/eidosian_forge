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
def test_get_routing_for_object():

    class Consumer(BaseEstimator):
        __metadata_request__fit = {'prop': None}
    assert_request_is_empty(get_routing_for_object(None))
    assert_request_is_empty(get_routing_for_object(object()))
    mr = MetadataRequest(owner='test')
    mr.fit.add_request(param='foo', alias='bar')
    mr_factory = get_routing_for_object(mr)
    assert_request_is_empty(mr_factory, exclude='fit')
    assert mr_factory.fit.requests == {'foo': 'bar'}
    mr = get_routing_for_object(Consumer())
    assert_request_is_empty(mr, exclude='fit')
    assert mr.fit.requests == {'prop': None}