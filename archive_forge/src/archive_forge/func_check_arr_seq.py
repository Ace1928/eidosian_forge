import itertools
import os
import sys
import tempfile
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arrays_equal
from ..array_sequence import ArraySequence, concatenate, is_array_sequence
def check_arr_seq(seq, arrays):
    lengths = list(map(len, arrays))
    assert is_array_sequence(seq)
    assert len(seq) == len(arrays)
    assert len(seq._offsets) == len(arrays)
    assert len(seq._lengths) == len(arrays)
    assert seq._data.shape[1:] == arrays[0].shape[1:]
    assert seq.common_shape == arrays[0].shape[1:]
    assert_arrays_equal(seq, arrays)
    if seq._is_view:
        assert_array_equal(sorted(seq._lengths), sorted(lengths))
    else:
        seq.shrink_data()
        assert seq._data.shape[0] == sum(lengths)
        assert_array_equal(seq._data, np.concatenate(arrays, axis=0))
        assert_array_equal(seq._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(seq._lengths, lengths)