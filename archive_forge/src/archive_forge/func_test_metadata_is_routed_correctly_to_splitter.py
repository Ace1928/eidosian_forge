import copy
import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.covariance import GraphicalLassoCV
from sklearn.ensemble import (
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.experimental import (
from sklearn.feature_selection import (
from sklearn.impute import IterativeImputer
from sklearn.linear_model import (
from sklearn.model_selection import (
from sklearn.multiclass import (
from sklearn.multioutput import (
from sklearn.pipeline import FeatureUnion
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tests.metadata_routing_common import (
from sklearn.utils.metadata_routing import MetadataRouter
@pytest.mark.parametrize('metaestimator', METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_metadata_is_routed_correctly_to_splitter(metaestimator):
    """Test that any requested metadata is correctly routed to the underlying
    splitters in CV estimators.
    """
    if 'cv_routing_methods' not in metaestimator:
        return
    cls = metaestimator['metaestimator']
    routing_methods = metaestimator['cv_routing_methods']
    X_ = metaestimator['X']
    y_ = metaestimator['y']
    for method_name in routing_methods:
        kwargs, (estimator, _), (scorer, _), (cv, registry) = get_init_args(metaestimator, sub_estimator_consumes=True)
        if estimator:
            estimator.set_fit_request(sample_weight=False, metadata=False)
        if scorer:
            scorer.set_score_request(sample_weight=False, metadata=False)
        cv.set_split_request(groups=True, metadata=True)
        instance = cls(**kwargs)
        method_kwargs = {'groups': groups, 'metadata': metadata}
        method = getattr(instance, method_name)
        method(X_, y_, **method_kwargs)
        assert registry
        for _splitter in registry:
            check_recorded_metadata(obj=_splitter, method='split', **method_kwargs)