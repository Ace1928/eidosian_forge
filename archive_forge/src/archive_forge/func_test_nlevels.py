import pytest
from rpy2.rinterface import NA_Character
from rpy2 import robjects
def test_nlevels():
    vec = robjects.FactorVector(robjects.StrVector('abaabc'))
    assert vec.nlevels == 3