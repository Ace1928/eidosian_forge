import pytest
import math
import rpy2.rinterface as ri
def test_NALogical_in_vector():
    na_bool = ri.NA_Logical
    x = ri.BoolSexpVector((True, na_bool, False))
    assert x[0] is True
    assert x[1] is ri.NA_Logical
    assert x[2] is False