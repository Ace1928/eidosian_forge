import importlib
import sys
import unittest
import warnings
from numbers import Integral, Real
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn import config_context, get_config
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning, SkipTestWarning
from sklearn.linear_model import (
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, NuSVC
from sklearn.utils import _array_api, all_estimators, deprecated
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
def test_check_estimator():
    msg = 'Passing a class was deprecated'
    with raises(TypeError, match=msg):
        check_estimator(object)
    msg = "Parameter 'p' of estimator 'HasMutableParameters' is of type object which is not allowed"
    check_estimator(HasImmutableParameters())
    with raises(AssertionError, match=msg):
        check_estimator(HasMutableParameters())
    msg = 'get_params result does not match what was passed to set_params'
    with raises(AssertionError, match=msg):
        check_estimator(ModifiesValueInsteadOfRaisingError())
    with warnings.catch_warnings(record=True) as records:
        check_estimator(RaisesErrorInSetParams())
    assert UserWarning in [rec.category for rec in records]
    with raises(AssertionError, match=msg):
        check_estimator(ModifiesAnotherValue())
    msg = "object has no attribute 'fit'"
    with raises(AttributeError, match=msg):
        check_estimator(BaseEstimator())
    msg = 'Did not raise'
    with raises(AssertionError, match=msg):
        check_estimator(BaseBadClassifier())
    try:
        from pandas import Series
        msg = "Estimator NoSampleWeightPandasSeriesType raises error if 'sample_weight' parameter is of type pandas.Series"
        with raises(ValueError, match=msg):
            check_estimator(NoSampleWeightPandasSeriesType())
    except ImportError:
        pass
    msg = "Estimator NoCheckinPredict doesn't check for NaN and inf in predict"
    with raises(AssertionError, match=msg):
        check_estimator(NoCheckinPredict())
    msg = 'Estimator changes __dict__ during predict'
    with raises(AssertionError, match=msg):
        check_estimator(ChangesDict())
    msg = 'Estimator ChangesWrongAttribute should not change or mutate  the parameter wrong_attribute from 0 to 1 during fit.'
    with raises(AssertionError, match=msg):
        check_estimator(ChangesWrongAttribute())
    check_estimator(ChangesUnderscoreAttribute())
    msg = 'Estimator adds public attribute\\(s\\) during the fit method. Estimators are only allowed to add private attributes either started with _ or ended with _ but wrong_attribute added'
    with raises(AssertionError, match=msg):
        check_estimator(SetsWrongAttribute())
    name = NotInvariantSampleOrder.__name__
    method = 'predict'
    msg = '{method} of {name} is not invariant when applied to a datasetwith different sample order.'.format(method=method, name=name)
    with raises(AssertionError, match=msg):
        check_estimator(NotInvariantSampleOrder())
    name = NotInvariantPredict.__name__
    method = 'predict'
    msg = '{method} of {name} is not invariant when applied to a subset.'.format(method=method, name=name)
    with raises(AssertionError, match=msg):
        check_estimator(NotInvariantPredict())
    name = NoSparseClassifier.__name__
    msg = "Estimator %s doesn't seem to fail gracefully on sparse data" % name
    with raises(AssertionError, match=msg):
        check_estimator(NoSparseClassifier())
    name = OneClassSampleErrorClassifier.__name__
    msg = f"{name} failed when fitted on one label after sample_weight trimming. Error message is not explicit, it should have 'class'."
    with raises(AssertionError, match=msg):
        check_estimator(OneClassSampleErrorClassifier())
    msg = "Estimator LargeSparseNotSupportedClassifier doesn't seem to support \\S{3}_64 matrix, and is not failing gracefully.*"
    with raises(AssertionError, match=msg):
        check_estimator(LargeSparseNotSupportedClassifier())
    msg = 'Only 2 classes are supported'
    with raises(ValueError, match=msg):
        check_estimator(UntaggedBinaryClassifier())
    for csr_container in CSR_CONTAINERS:
        check_estimator(SparseTransformer(sparse_container=csr_container))
    check_estimator(LogisticRegression())
    check_estimator(LogisticRegression(C=0.01))
    check_estimator(MultiTaskElasticNet())
    check_estimator(TaggedBinaryClassifier())
    check_estimator(RequiresPositiveXRegressor())
    msg = 'negative y values not supported!'
    with raises(ValueError, match=msg):
        check_estimator(RequiresPositiveYRegressor())
    check_estimator(PoorScoreLogisticRegression())