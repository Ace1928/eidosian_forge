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
def test_error_on_missing_requests_for_sub_estimator(metaestimator):
    if 'estimator' not in metaestimator:
        return
    cls = metaestimator['metaestimator']
    X = metaestimator['X']
    y = metaestimator['y']
    routing_methods = metaestimator['estimator_routing_methods']
    for method_name in routing_methods:
        for key in ['sample_weight', 'metadata']:
            kwargs, (estimator, _), (scorer, _), *_ = get_init_args(metaestimator, sub_estimator_consumes=True)
            if scorer:
                scorer.set_score_request(**{key: True})
            val = {'sample_weight': sample_weight, 'metadata': metadata}[key]
            method_kwargs = {key: val}
            msg = f'[{key}] are passed but are not explicitly set as requested or not for {estimator.__class__.__name__}.{method_name}'
            instance = cls(**kwargs)
            with pytest.raises(UnsetMetadataPassedError, match=re.escape(msg)):
                method = getattr(instance, method_name)
                method(X, y, **method_kwargs)