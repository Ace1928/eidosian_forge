import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_nalogical():
    vec = robjects.BoolVector((True, False, True))
    vec[0] = robjects.NA_Logical
    assert robjects.baseenv['is.na'](vec)[0] is True