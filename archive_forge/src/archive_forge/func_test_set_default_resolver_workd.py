from ..resolver import (
def test_set_default_resolver_workd():
    default_resolver = get_default_resolver()
    set_default_resolver(dict_resolver)
    assert get_default_resolver() == dict_resolver
    set_default_resolver(default_resolver)