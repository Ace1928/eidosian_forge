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
@pytest.mark.parametrize('constructor', [list, np.array])
def test_graphviz_feature_class_names_array_support(constructor):
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion='gini', random_state=2)
    clf.fit(X, y)
    contents1 = export_graphviz(clf, feature_names=constructor(['feature0', 'feature1']), out_file=None)
    contents2 = 'digraph Tree {\nnode [shape=box, fontname="helvetica"] ;\nedge [fontname="helvetica"] ;\n0 [label="feature0 <= 0.0\\ngini = 0.5\\nsamples = 6\\nvalue = [3, 3]"] ;\n1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n}'
    assert contents1 == contents2
    contents1 = export_graphviz(clf, class_names=constructor(['yes', 'no']), out_file=None)
    contents2 = 'digraph Tree {\nnode [shape=box, fontname="helvetica"] ;\nedge [fontname="helvetica"] ;\n0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\nvalue = [3, 3]\\nclass = yes"] ;\n1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\nclass = yes"] ;\n0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\nclass = no"] ;\n0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n}'
    assert contents1 == contents2