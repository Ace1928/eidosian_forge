from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_exact_uri(self):
    resource = Resource.opaque(contents={'foo': 'baz'})
    resolver = Registry({'http://example.com/1': resource}).resolver()
    resolved = resolver.lookup('http://example.com/1')
    assert resolved.contents == resource.contents