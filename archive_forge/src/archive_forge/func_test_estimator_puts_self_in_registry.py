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
@pytest.mark.parametrize('estimator', [ConsumingClassifier(registry=_Registry()), ConsumingRegressor(registry=_Registry()), ConsumingTransformer(registry=_Registry()), WeightedMetaClassifier(estimator=ConsumingClassifier(), registry=_Registry()), WeightedMetaRegressor(estimator=ConsumingRegressor(), registry=_Registry())])
def test_estimator_puts_self_in_registry(estimator):
    """Check that an estimator puts itself in the registry upon fit."""
    estimator.fit(X, y)
    assert estimator in estimator.registry