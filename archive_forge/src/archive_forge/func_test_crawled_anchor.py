from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_crawled_anchor(self):
    resource = ID_AND_CHILDREN.create_resource({'anchors': {'foo': 'bar'}})
    registry = Registry().with_resource('urn:example', resource)
    retrieved = registry.anchor('urn:example', 'foo')
    assert retrieved.value == Anchor(name='foo', resource=ID_AND_CHILDREN.create_resource('bar'))
    assert retrieved.registry == registry.crawl()