from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_anchor_without_id(self):
    root = ID_AND_CHILDREN.create_resource({'anchors': {'foo': 12}})
    resolver = Registry().with_resource('urn:example', root).resolver()
    resolved = resolver.lookup('urn:example#foo')
    assert resolved.contents == 12