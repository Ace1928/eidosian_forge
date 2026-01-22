import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_setitem():
    vec = robjects.r.seq(1, 10)
    assert vec[0] == 1
    vec[0] = 20
    assert vec[0] == 20