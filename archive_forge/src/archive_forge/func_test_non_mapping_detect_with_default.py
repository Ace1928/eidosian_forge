from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_non_mapping_detect_with_default(self):
    specification = ID_AND_CHILDREN.detect(True)
    assert specification is ID_AND_CHILDREN