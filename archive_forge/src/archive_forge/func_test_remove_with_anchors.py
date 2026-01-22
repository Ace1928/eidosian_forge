from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_remove_with_anchors(self):
    one = Resource.opaque(contents={})
    two = ID_AND_CHILDREN.create_resource({'anchors': {'foo': 'bar'}})
    registry = Registry().with_resources([('urn:foo', one), ('urn:bar', two)]).crawl()
    assert registry.remove('urn:bar') == Registry().with_resource('urn:foo', one).crawl()