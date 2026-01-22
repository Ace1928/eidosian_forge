import pickle
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from io import StringIO
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import _IS_WASM, IS_PYPY
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_strip_accents():
    a = 'àáâãäåçèéêë'
    expected = 'aaaaaaceeee'
    assert strip_accents_unicode(a) == expected
    a = 'ìíîïñòóôõöùúûüý'
    expected = 'iiiinooooouuuuy'
    assert strip_accents_unicode(a) == expected
    a = 'إ'
    expected = 'ا'
    assert strip_accents_unicode(a) == expected
    a = 'this is à test'
    expected = 'this is a test'
    assert strip_accents_unicode(a) == expected
    a = 'ö'
    expected = 'o'
    assert strip_accents_unicode(a) == expected
    a = '̀́̂̃'
    expected = ''
    assert strip_accents_unicode(a) == expected
    a = 'ȫ'
    expected = 'o'
    assert strip_accents_unicode(a) == expected