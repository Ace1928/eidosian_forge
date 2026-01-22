from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_detect_from_json_schema(self):
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
    specification = Specification.detect(schema)
    assert specification == DRAFT202012