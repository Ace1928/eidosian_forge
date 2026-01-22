from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_detect_with_fallback(self):
    specification = Specification.OPAQUE.detect({'foo': 'bar'})
    assert specification is Specification.OPAQUE