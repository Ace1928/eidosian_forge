import re
from pprint import PrettyPrinter
import numpy as np
from sklearn.utils._pprint import _EstimatorPrettyPrinter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import config_context
def test_changed_only():
    lr = LogisticRegression(C=99)
    expected = 'LogisticRegression(C=99)'
    assert lr.__repr__() == expected
    lr = LogisticRegression(C=99, class_weight=0.4, fit_intercept=False, tol=1234, verbose=True)
    expected = '\nLogisticRegression(C=99, class_weight=0.4, fit_intercept=False, tol=1234,\n                   verbose=True)'
    expected = expected[1:]
    assert lr.__repr__() == expected
    imputer = SimpleImputer(missing_values=0)
    expected = 'SimpleImputer(missing_values=0)'
    assert imputer.__repr__() == expected
    imputer = SimpleImputer(missing_values=float('NaN'))
    expected = 'SimpleImputer()'
    assert imputer.__repr__() == expected
    repr(LogisticRegressionCV(Cs=np.array([0.1, 1])))