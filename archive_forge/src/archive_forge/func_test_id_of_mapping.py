import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
@pytest.mark.parametrize('id, specification', [('$id', referencing.jsonschema.DRAFT202012), ('$id', referencing.jsonschema.DRAFT201909), ('$id', referencing.jsonschema.DRAFT7), ('$id', referencing.jsonschema.DRAFT6), ('id', referencing.jsonschema.DRAFT4), ('id', referencing.jsonschema.DRAFT3)])
def test_id_of_mapping(id, specification):
    uri = 'http://example.com/some-schema'
    assert specification.id_of({id: uri}) == uri