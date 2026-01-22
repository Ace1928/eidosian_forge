from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_pointer_to_empty_string(self):
    resolver = Registry().resolver_with_root(Resource.opaque({'': {}}))
    assert resolver.lookup('#/').contents == {}