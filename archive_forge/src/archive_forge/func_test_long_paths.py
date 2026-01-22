import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_long_paths(self):
    long_test1 = self.build_operands('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
    path, path_str = np.einsum_path(*long_test1, optimize='greedy')
    self.assert_path_equal(path, ['einsum_path', (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])
    path, path_str = np.einsum_path(*long_test1, optimize='optimal')
    self.assert_path_equal(path, ['einsum_path', (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])
    long_test2 = self.build_operands('chd,bde,agbc,hiad,bdi,cgh,agdb')
    path, path_str = np.einsum_path(*long_test2, optimize='greedy')
    self.assert_path_equal(path, ['einsum_path', (3, 4), (0, 3), (3, 4), (1, 3), (1, 2), (0, 1)])
    path, path_str = np.einsum_path(*long_test2, optimize='optimal')
    self.assert_path_equal(path, ['einsum_path', (0, 5), (1, 4), (3, 4), (1, 3), (1, 2), (0, 1)])