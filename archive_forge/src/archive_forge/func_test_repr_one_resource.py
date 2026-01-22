from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_repr_one_resource(self):
    registry = Registry().with_resource(uri='http://example.com/1', resource=Resource.opaque(contents={}))
    assert repr(registry) == '<Registry (1 uncrawled resource)>'