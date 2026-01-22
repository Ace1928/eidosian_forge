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
def test_default_request(metaestimator):
    cls = metaestimator['metaestimator']
    kwargs, *_ = get_init_args(metaestimator, sub_estimator_consumes=True)
    instance = cls(**kwargs)
    if 'cv_name' in metaestimator:
        exclude = {'splitter': ['split']}
    else:
        exclude = None
    assert_request_is_empty(instance.get_metadata_routing(), exclude=exclude)
    assert isinstance(instance.get_metadata_routing(), MetadataRouter)