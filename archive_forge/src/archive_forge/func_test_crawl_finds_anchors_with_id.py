from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_crawl_finds_anchors_with_id(self):
    resource = ID_AND_CHILDREN.create_resource({'ID': 'urn:bar', 'anchors': {'foo': 12}})
    registry = resource @ Registry()
    assert registry.crawl().anchor(resource.id(), 'foo').value == Anchor(name='foo', resource=ID_AND_CHILDREN.create_resource(12))