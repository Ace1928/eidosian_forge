from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_resolver_with_root_identified(self):
    root = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com'})
    resolver = Registry().resolver_with_root(root)
    assert resolver.lookup('http://example.com').contents == root.contents
    assert resolver.lookup('#').contents == root.contents