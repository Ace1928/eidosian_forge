import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_seq_repeat(self):
    accepted_types = set(np.typecodes['AllInteger'])
    deprecated_types = {'?'}
    forbidden_types = set(np.typecodes['All']) - accepted_types - deprecated_types
    forbidden_types -= {'V'}
    for seq_type in (list, tuple):
        seq = seq_type([1, 2, 3])
        for numpy_type in accepted_types:
            i = np.dtype(numpy_type).type(2)
            assert_equal(seq * i, seq * int(i))
            assert_equal(i * seq, int(i) * seq)
        for numpy_type in deprecated_types:
            i = np.dtype(numpy_type).type()
            assert_equal(assert_warns(DeprecationWarning, operator.mul, seq, i), seq * int(i))
            assert_equal(assert_warns(DeprecationWarning, operator.mul, i, seq), int(i) * seq)
        for numpy_type in forbidden_types:
            i = np.dtype(numpy_type).type()
            assert_raises(TypeError, operator.mul, seq, i)
            assert_raises(TypeError, operator.mul, i, seq)