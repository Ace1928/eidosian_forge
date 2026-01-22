from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_from_contents_unneeded_default(self):
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
    resource = Resource.from_contents(schema, default_specification=Specification.OPAQUE)
    assert resource == Resource(contents=schema, specification=DRAFT202012)