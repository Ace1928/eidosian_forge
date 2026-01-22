from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_pointer_to_mapping(self):
    resource = Resource.opaque(contents={'foo': 'baz'})
    resolver = Registry().resolver()
    assert resource.pointer('/foo', resolver=resolver).contents == 'baz'