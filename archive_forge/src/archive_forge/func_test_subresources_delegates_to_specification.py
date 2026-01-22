from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_subresources_delegates_to_specification(self):
    resource = ID_AND_CHILDREN.create_resource({'children': [{}, 12]})
    assert list(resource.subresources()) == [ID_AND_CHILDREN.create_resource(each) for each in [{}, 12]]