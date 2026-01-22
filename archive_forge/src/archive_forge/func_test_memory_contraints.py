import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_memory_contraints(self):
    outer_test = self.build_operands('a,b,c->abc')
    path, path_str = np.einsum_path(*outer_test, optimize=('greedy', 0))
    self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])
    path, path_str = np.einsum_path(*outer_test, optimize=('optimal', 0))
    self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])
    long_test = self.build_operands('acdf,jbje,gihb,hfac')
    path, path_str = np.einsum_path(*long_test, optimize=('greedy', 0))
    self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])
    path, path_str = np.einsum_path(*long_test, optimize=('optimal', 0))
    self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])