import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_size_by_dict():
    sizes_dict = {}
    for ind, val in zip('abcdez', [2, 5, 9, 11, 13, 0]):
        sizes_dict[ind] = val
    path_func = oe.helpers.compute_size_by_dict
    assert 1 == path_func('', sizes_dict)
    assert 2 == path_func('a', sizes_dict)
    assert 5 == path_func('b', sizes_dict)
    assert 0 == path_func('z', sizes_dict)
    assert 0 == path_func('az', sizes_dict)
    assert 0 == path_func('zbc', sizes_dict)
    assert 104 == path_func('aaae', sizes_dict)
    assert 12870 == path_func('abcde', sizes_dict)