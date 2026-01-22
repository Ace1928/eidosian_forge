from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_repr_mixed_crawled(self):
    one = Resource.opaque(contents={})
    two = ID_AND_CHILDREN.create_resource({'foo': 'bar'})
    registry = Registry({'http://example.com/1': one}).crawl().with_resource(uri='http://example.com/foo/bar', resource=two)
    assert repr(registry) == '<Registry (2 resources, 1 uncrawled)>'