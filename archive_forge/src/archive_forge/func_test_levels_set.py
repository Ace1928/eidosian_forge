import pytest
from rpy2.rinterface import NA_Character
from rpy2 import robjects
def test_levels_set():
    vec = robjects.FactorVector(robjects.StrVector('abaabc'))
    vec.levels = robjects.vectors.StrVector('def')
    assert set(('d', 'e', 'f')) == set(tuple(vec.levels))