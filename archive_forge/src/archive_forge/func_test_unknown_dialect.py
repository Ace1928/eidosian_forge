import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_unknown_dialect():
    dialect_id = 'http://example.com/unknown-json-schema-dialect-id'
    with pytest.raises(referencing.jsonschema.UnknownDialect) as excinfo:
        Resource.from_contents({'$schema': dialect_id})
    assert excinfo.value.uri == dialect_id