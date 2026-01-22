from ..resolver import (
def test_attr_resolver():
    resolved = attr_resolver('attr', None, demo_obj, info, **args)
    assert resolved == 'value'