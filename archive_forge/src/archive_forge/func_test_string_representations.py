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
@pytest.mark.parametrize('obj, string', [(MethodMetadataRequest(owner='test', method='fit').add_request(param='foo', alias='bar'), "{'foo': 'bar'}"), (MetadataRequest(owner='test'), '{}'), (MethodMapping.from_str('score'), "[{'callee': 'score', 'caller': 'score'}]"), (MetadataRouter(owner='test').add(method_mapping='predict', estimator=ConsumingRegressor()), "{'estimator': {'mapping': [{'callee': 'predict', 'caller': 'predict'}], 'router': {'fit': {'sample_weight': None, 'metadata': None}, 'partial_fit': {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight': None, 'metadata': None}, 'score': {'sample_weight': None}}}}")])
def test_string_representations(obj, string):
    assert str(obj) == string