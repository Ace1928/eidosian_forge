import re
from pytest import raises
from graphql import parse, get_introspection_query, validate
from ...types import Schema, ObjectType, Interface
from ...types import String, Int, List, Field
from ..depth_limit import depth_limit_validator
def test_should_ignore_the_introspection_query():
    errors, result = run_query(get_introspection_query(), 10)
    assert not errors
    assert result == {'IntrospectionQuery': 0}