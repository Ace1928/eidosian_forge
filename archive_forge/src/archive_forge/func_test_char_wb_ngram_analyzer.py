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
def test_char_wb_ngram_analyzer():
    cnga = CountVectorizer(analyzer='char_wb', strip_accents='unicode', ngram_range=(3, 6)).build_analyzer()
    text = 'This \n\tis a test, really.\n\n I met Harry yesterday'
    expected = [' th', 'thi', 'his', 'is ', ' thi']
    assert cnga(text)[:5] == expected
    expected = ['yester', 'esterd', 'sterda', 'terday', 'erday ']
    assert cnga(text)[-5:] == expected
    cnga = CountVectorizer(input='file', analyzer='char_wb', ngram_range=(3, 6)).build_analyzer()
    text = StringIO('A test with a file-like object!')
    expected = [' a ', ' te', 'tes', 'est', 'st ', ' tes']
    assert cnga(text)[:6] == expected