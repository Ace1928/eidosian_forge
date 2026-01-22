from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_pointer_to_array(self):
    resource = Resource.opaque(contents={'foo': {'bar': [3]}})
    resolver = Registry().resolver()
    assert resource.pointer('/foo/bar/0', resolver=resolver).contents == 3