from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_contents_strips_empty_fragments(self):
    uri = 'http://example.com/'
    resource = ID_AND_CHILDREN.create_resource({'ID': uri + '#'})
    registry = resource @ Registry()
    assert registry.contents(uri) == registry.contents(uri + '#') == {'ID': uri + '#'}