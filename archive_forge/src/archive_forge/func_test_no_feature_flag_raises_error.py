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
def test_no_feature_flag_raises_error():
    """Test that when feature flag disabled, set_{method}_requests raises."""
    with config_context(enable_metadata_routing=False):
        with pytest.raises(RuntimeError, match='This method is only available'):
            ConsumingClassifier().set_fit_request(sample_weight=True)