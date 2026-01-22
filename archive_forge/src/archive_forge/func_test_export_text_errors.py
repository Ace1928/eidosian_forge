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
def test_export_text_errors():
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    err_msg = 'feature_names must contain 2 elements, got 1'
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, feature_names=['a'])
    err_msg = 'When `class_names` is an array, it should contain as many items as `decision_tree.classes_`. Got 1 while the tree was fitted with 2 classes.'
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, class_names=['a'])