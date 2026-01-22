import pytest
import math
import rpy2.rinterface as ri
def test_NAInteger_repr():
    na = ri.NA_Integer
    assert repr(na) == 'NA_integer_'