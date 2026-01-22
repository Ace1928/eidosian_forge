import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_empty_registry():
    assert referencing.jsonschema.EMPTY_REGISTRY == Registry()