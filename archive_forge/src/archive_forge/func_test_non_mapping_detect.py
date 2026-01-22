from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_non_mapping_detect(self):
    with pytest.raises(exceptions.CannotDetermineSpecification):
        Specification.detect(True)