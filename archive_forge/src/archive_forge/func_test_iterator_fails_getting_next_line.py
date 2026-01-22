import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_iterator_fails_getting_next_line():

    class BadSequence:

        def __len__(self):
            return 100

        def __getitem__(self, item):
            if item == 50:
                raise RuntimeError('Bad things happened!')
            return f'{item}, {item + 1}'
    with pytest.raises(RuntimeError, match='Bad things happened!'):
        np.loadtxt(BadSequence(), dtype=int, delimiter=',')