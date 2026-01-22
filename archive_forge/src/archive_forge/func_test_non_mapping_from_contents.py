from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_non_mapping_from_contents(self):
    resource = Resource.from_contents(True, default_specification=ID_AND_CHILDREN)
    assert resource == Resource(contents=True, specification=ID_AND_CHILDREN)