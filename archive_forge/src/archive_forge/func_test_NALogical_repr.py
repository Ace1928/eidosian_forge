import pytest
import math
import rpy2.rinterface as ri
def test_NALogical_repr():
    na = ri.NA_Logical
    assert repr(na) == 'NA'