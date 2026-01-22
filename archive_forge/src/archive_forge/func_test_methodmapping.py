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
def test_methodmapping():
    mm = MethodMapping().add(caller='fit', callee='transform').add(caller='fit', callee='fit')
    mm_list = list(mm)
    assert mm_list[0] == ('transform', 'fit')
    assert mm_list[1] == ('fit', 'fit')
    mm = MethodMapping.from_str('one-to-one')
    for method in METHODS:
        assert MethodPair(method, method) in mm._routes
    assert len(mm._routes) == len(METHODS)
    mm = MethodMapping.from_str('score')
    assert repr(mm) == "[{'callee': 'score', 'caller': 'score'}]"