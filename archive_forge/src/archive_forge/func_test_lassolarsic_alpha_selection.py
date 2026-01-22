import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
@pytest.mark.parametrize('criterion', ['aic', 'bic'])
def test_lassolarsic_alpha_selection(criterion):
    """Check that we properly compute the AIC and BIC score.

    In this test, we reproduce the example of the Fig. 2 of Zou et al.
    (reference [1] in LassoLarsIC) In this example, only 7 features should be
    selected.
    """
    model = make_pipeline(StandardScaler(), LassoLarsIC(criterion=criterion))
    model.fit(X, y)
    best_alpha_selected = np.argmin(model[-1].criterion_)
    assert best_alpha_selected == 7