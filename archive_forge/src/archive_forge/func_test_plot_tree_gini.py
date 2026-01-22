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
def test_plot_tree_gini(pyplot):
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion='gini', random_state=2)
    clf.fit(X, y)
    feature_names = ['first feat', 'sepal_width']
    nodes = plot_tree(clf, feature_names=feature_names)
    assert len(nodes) == 3
    assert nodes[0].get_text() == 'first feat <= 0.0\ngini = 0.5\nsamples = 6\nvalue = [3, 3]'
    assert nodes[1].get_text() == 'gini = 0.0\nsamples = 3\nvalue = [3, 0]'
    assert nodes[2].get_text() == 'gini = 0.0\nsamples = 3\nvalue = [0, 3]'