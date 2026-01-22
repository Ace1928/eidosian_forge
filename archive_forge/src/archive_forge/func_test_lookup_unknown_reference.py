from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_unknown_reference(self):
    resolver = Registry().resolver()
    ref = 'http://example.com/does/not/exist'
    with pytest.raises(exceptions.Unresolvable) as e:
        resolver.lookup(ref)
    assert e.value == exceptions.Unresolvable(ref=ref)