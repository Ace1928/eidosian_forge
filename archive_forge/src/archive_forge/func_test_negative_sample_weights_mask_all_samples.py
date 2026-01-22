import re
import numpy as np
import pytest
from numpy.testing import (
from sklearn import base, datasets, linear_model, metrics, svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import (
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import (  # type: ignore
from sklearn.svm._classes import _validate_dual_parameter
from sklearn.utils import check_random_state, shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('Estimator, err_msg', [(svm.SVC, 'Invalid input - all samples have zero or negative weights.'), (svm.NuSVC, '(negative dimensions are not allowed|nu is infeasible)'), (svm.SVR, 'Invalid input - all samples have zero or negative weights.'), (svm.NuSVR, 'Invalid input - all samples have zero or negative weights.'), (svm.OneClassSVM, 'Invalid input - all samples have zero or negative weights.')], ids=['SVC', 'NuSVC', 'SVR', 'NuSVR', 'OneClassSVM'])
@pytest.mark.parametrize('sample_weight', [[0] * len(Y), [-0.3] * len(Y)], ids=['weights-are-zero', 'weights-are-negative'])
def test_negative_sample_weights_mask_all_samples(Estimator, err_msg, sample_weight):
    est = Estimator(kernel='linear')
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, Y, sample_weight=sample_weight)