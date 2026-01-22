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
def test_nested_routing():
    pipeline = SimplePipeline([MetaTransformer(transformer=ConsumingTransformer().set_fit_request(metadata=True, sample_weight=False).set_transform_request(sample_weight=True, metadata=False)), WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight='inner_weights', metadata=False).set_predict_request(sample_weight=False)).set_fit_request(sample_weight='outer_weights')])
    w1, w2, w3 = ([1], [2], [3])
    pipeline.fit(X, y, metadata=my_groups, sample_weight=w1, outer_weights=w2, inner_weights=w3)
    check_recorded_metadata(pipeline.steps_[0].transformer_, 'fit', metadata=my_groups, sample_weight=None)
    check_recorded_metadata(pipeline.steps_[0].transformer_, 'transform', sample_weight=w1, metadata=None)
    check_recorded_metadata(pipeline.steps_[1], 'fit', sample_weight=w2)
    check_recorded_metadata(pipeline.steps_[1].estimator_, 'fit', sample_weight=w3)
    pipeline.predict(X, sample_weight=w3)
    check_recorded_metadata(pipeline.steps_[0].transformer_, 'transform', sample_weight=w3, metadata=None)