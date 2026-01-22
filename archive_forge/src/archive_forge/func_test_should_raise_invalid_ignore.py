import re
from pytest import raises
from graphql import parse, get_introspection_query, validate
from ...types import Schema, ObjectType, Interface
from ...types import String, Int, List, Field
from ..depth_limit import depth_limit_validator
def test_should_raise_invalid_ignore():
    query = '\n    query read1 {\n      user { address { city } }\n    }\n    '
    with raises(ValueError, match='Invalid ignore option:'):
        run_query(query, 10, ignore=[True])