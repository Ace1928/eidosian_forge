from io import StringIO
from re import finditer, search
from textwrap import dedent
import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import (
def test_friedman_mse_in_graphviz():
    clf = DecisionTreeRegressor(criterion='friedman_mse', random_state=0)
    clf.fit(X, y)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data)
    clf = GradientBoostingClassifier(n_estimators=2, random_state=0)
    clf.fit(X, y)
    for estimator in clf.estimators_:
        export_graphviz(estimator[0], out_file=dot_data)
    for finding in finditer('\\[.*?samples.*?\\]', dot_data.getvalue()):
        assert 'friedman_mse' in finding.group()