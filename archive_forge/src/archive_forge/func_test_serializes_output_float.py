from graphql import Undefined
from ..scalars import Boolean, Float, Int, String
def test_serializes_output_float():
    assert Float.serialize(1) == 1.0
    assert Float.serialize(0) == 0.0
    assert Float.serialize(-1) == -1.0
    assert Float.serialize(0.1) == 0.1
    assert Float.serialize(1.1) == 1.1
    assert Float.serialize(-1.1) == -1.1
    assert Float.serialize('-1.1') == -1.1
    assert Float.serialize('one') is Undefined
    assert Float.serialize(False) == 0
    assert Float.serialize(True) == 1