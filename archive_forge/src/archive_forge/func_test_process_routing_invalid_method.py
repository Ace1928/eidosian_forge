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
def test_process_routing_invalid_method():
    with pytest.raises(TypeError, match='Can only route and process input'):
        process_routing(ConsumingClassifier(), 'invalid_method', groups=my_groups)