import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_broadcasting_dot_cases(self):
    a = np.random.rand(1, 5, 4)
    b = np.random.rand(4, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(10)
    self.optimize_compare('ijk,kl,jl', operands=[a, b, c])
    self.optimize_compare('ijk,kl,jl,i->i', operands=[a, b, c, d])
    e = np.random.rand(1, 1, 5, 4)
    f = np.random.rand(7, 7)
    self.optimize_compare('abjk,kl,jl', operands=[e, b, c])
    self.optimize_compare('abjk,kl,jl,ab->ab', operands=[e, b, c, f])
    g = np.arange(64).reshape(2, 4, 8)
    self.optimize_compare('obk,ijk->ioj', operands=[g, g])