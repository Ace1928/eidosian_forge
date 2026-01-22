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
def test_metadata_request_consumes_method():
    """Test that MetadataRequest().consumes() method works as expected."""
    request = MetadataRouter(owner='test')
    assert request.consumes(method='fit', params={'foo'}) == set()
    request = MetadataRequest(owner='test')
    request.fit.add_request(param='foo', alias=True)
    assert request.consumes(method='fit', params={'foo'}) == {'foo'}
    request = MetadataRequest(owner='test')
    request.fit.add_request(param='foo', alias='bar')
    assert request.consumes(method='fit', params={'bar', 'foo'}) == {'bar'}