import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_strvector_factor():
    vec = robjects.StrVector(('abc', 'def', 'abc'))
    fvec = vec.factor()
    assert isinstance(fvec, robjects.FactorVector)