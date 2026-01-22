import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
@pytest.mark.parametrize('specification', [referencing.jsonschema.DRAFT202012, referencing.jsonschema.DRAFT201909, referencing.jsonschema.DRAFT7, referencing.jsonschema.DRAFT6])
@pytest.mark.parametrize('value', [True, False])
def test_subresources_of_bool(specification, value):
    assert list(specification.subresources_of(value)) == []