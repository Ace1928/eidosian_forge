from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_crawl_still_has_top_level_resource(self):
    resource = Resource.opaque({'foo': 'bar'})
    uri = 'urn:example'
    registry = Registry({uri: resource}).crawl()
    assert registry[uri] is resource