import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_broken_sequence_not_nd_index(self):

    class SequenceLike:

        def __index__(self):
            return 0

        def __len__(self):
            return 1

        def __getitem__(self, item):
            raise IndexError('Not possible')
    arr = np.arange(10)
    assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])
    arr = np.zeros((1,), dtype=[('f1', 'i8'), ('f2', 'i8')])
    assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])