import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_specification_with_unknown_dialect():
    dialect_id = 'http://example.com/unknown-json-schema-dialect-id'
    with pytest.raises(referencing.jsonschema.UnknownDialect) as excinfo:
        referencing.jsonschema.specification_with(dialect_id)
    assert excinfo.value.uri == dialect_id