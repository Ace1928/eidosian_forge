from graphql import Undefined
from ..scalars import Boolean, Float, Int, String
def test_serializes_output_string():
    assert String.serialize('string') == 'string'
    assert String.serialize(1) == '1'
    assert String.serialize(-1.1) == '-1.1'
    assert String.serialize(True) == 'true'
    assert String.serialize(False) == 'false'
    assert String.serialize('😁') == '😁'