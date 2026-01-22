from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_anchor_in_nonexistent_resource(self):
    registry = Registry()
    with pytest.raises(exceptions.NoSuchResource) as e:
        registry.anchor('urn:example', 'foo')
    assert e.value == exceptions.NoSuchResource(ref='urn:example')