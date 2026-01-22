import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_sample_probabilities():
    vec = robjects.IntVector(range(100))
    spl = vec.sample(10, probabilities=robjects.FloatVector([0.01] * 100))
    assert len(spl) == 10