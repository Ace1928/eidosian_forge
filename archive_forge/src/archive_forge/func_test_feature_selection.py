from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
def test_feature_selection():
    d1 = dict([('useless%d' % i, 10) for i in range(20)], useful1=1, useful2=20)
    d2 = dict([('useless%d' % i, 10) for i in range(20)], useful1=20, useful2=1)
    for indices in (True, False):
        v = DictVectorizer().fit([d1, d2])
        X = v.transform([d1, d2])
        sel = SelectKBest(chi2, k=2).fit(X, [0, 1])
        v.restrict(sel.get_support(indices=indices), indices=indices)
        assert_array_equal(v.get_feature_names_out(), ['useful1', 'useful2'])