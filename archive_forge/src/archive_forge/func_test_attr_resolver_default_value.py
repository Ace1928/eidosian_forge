from ..resolver import (
def test_attr_resolver_default_value():
    resolved = attr_resolver('attr2', 'default', demo_obj, info, **args)
    assert resolved == 'default'