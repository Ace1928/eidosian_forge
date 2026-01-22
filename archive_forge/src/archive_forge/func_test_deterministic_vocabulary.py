from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
def test_deterministic_vocabulary(global_random_seed):
    items = [('%03d' % i, i) for i in range(1000)]
    rng = Random(global_random_seed)
    d_sorted = dict(items)
    rng.shuffle(items)
    d_shuffled = dict(items)
    v_1 = DictVectorizer().fit([d_sorted])
    v_2 = DictVectorizer().fit([d_shuffled])
    assert v_1.vocabulary_ == v_2.vocabulary_