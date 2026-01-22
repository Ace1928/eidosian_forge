from ..resolver import (
def test_dict_or_attr_resolver():
    resolved = dict_or_attr_resolver('attr', None, demo_dict, info, **args)
    assert resolved == 'value'
    resolved = dict_or_attr_resolver('attr', None, demo_obj, info, **args)
    assert resolved == 'value'