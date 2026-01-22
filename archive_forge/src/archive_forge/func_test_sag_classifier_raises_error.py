import math
import re
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import make_dataset
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.linear_model._sag import get_auto_step_size
from sklearn.linear_model._sag_fast import _multinomial_grad_loss_all_samples
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('solver', ['sag', 'saga'])
def test_sag_classifier_raises_error(solver):
    rng = np.random.RandomState(42)
    X, y = make_classification(random_state=rng)
    clf = LogisticRegression(solver=solver, random_state=rng, warm_start=True)
    clf.fit(X, y)
    clf.coef_[:] = np.nan
    with pytest.raises(ValueError, match='Floating-point under-/overflow'):
        clf.fit(X, y)