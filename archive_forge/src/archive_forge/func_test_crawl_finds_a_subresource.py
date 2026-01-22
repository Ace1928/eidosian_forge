from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_crawl_finds_a_subresource(self):
    child_id = 'urn:child'
    root = ID_AND_CHILDREN.create_resource({'ID': 'urn:root', 'children': [{'ID': child_id, 'foo': 12}]})
    registry = root @ Registry()
    with pytest.raises(LookupError):
        registry[child_id]
    expected = ID_AND_CHILDREN.create_resource({'ID': child_id, 'foo': 12})
    assert registry.crawl()[child_id] == expected