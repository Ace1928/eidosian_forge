import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_r_truediv():
    v = robjects.vectors.IntVector((2, 3, 4))
    res = v.ro / 2
    assert all((abs(x - y) < 0.001 for x, y in zip(res, (1, 1.5, 2))))