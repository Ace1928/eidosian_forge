import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.usefixtures('enable_slep006')
@pytest.mark.parametrize('MultiTaskEstimatorCV', [MultiTaskElasticNetCV, MultiTaskLassoCV])
def test_multitask_cv_estimators_with_sample_weight(MultiTaskEstimatorCV):
    """Check that for :class:`MultiTaskElasticNetCV` and
    class:`MultiTaskLassoCV` if `sample_weight` is passed and the
    CV splitter does not support `sample_weight` an error is raised.
    On the other hand if the splitter does support `sample_weight`
    while `sample_weight` is passed there is no error and process
    completes smoothly as before.
    """

    class CVSplitter(BaseCrossValidator, GroupsConsumerMixin):

        def get_n_splits(self, X=None, y=None, groups=None, metadata=None):
            pass

    class CVSplitterSampleWeight(CVSplitter):

        def split(self, X, y=None, groups=None, sample_weight=None):
            split_index = len(X) // 2
            train_indices = list(range(0, split_index))
            test_indices = list(range(split_index, len(X)))
            yield (test_indices, train_indices)
            yield (train_indices, test_indices)
    X, y = make_regression(random_state=42, n_targets=2)
    sample_weight = np.ones(X.shape[0])
    splitter = CVSplitter().set_split_request(groups=True)
    estimator = MultiTaskEstimatorCV(cv=splitter)
    msg = 'do not support sample weights'
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y, sample_weight=sample_weight)
    splitter = CVSplitterSampleWeight().set_split_request(groups=True, sample_weight=True)
    estimator = MultiTaskEstimatorCV(cv=splitter)
    estimator.fit(X, y, sample_weight=sample_weight)