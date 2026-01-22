from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_remove_uncrawled(self):
    one = Resource.opaque(contents={})
    two = ID_AND_CHILDREN.create_resource({'foo': 'bar'})
    registry = Registry().with_resources([('urn:foo', one), ('urn:bar', two)])
    assert registry.remove('urn:foo') == Registry().with_resource('urn:bar', two)