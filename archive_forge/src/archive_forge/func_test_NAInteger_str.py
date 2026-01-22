import pytest
import math
import rpy2.rinterface as ri
def test_NAInteger_str():
    na = ri.NA_Integer
    assert str(na) == 'NA_integer_'