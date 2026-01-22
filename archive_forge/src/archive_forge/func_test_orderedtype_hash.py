from ..orderedtype import OrderedType
def test_orderedtype_hash():
    one = OrderedType()
    two = OrderedType()
    assert hash(one) == hash(one)
    assert hash(one) != hash(two)