import re
from pytest import raises
from graphql import parse, get_introspection_query, validate
from ...types import Schema, ObjectType, Interface
from ...types import String, Int, List, Field
from ..depth_limit import depth_limit_validator
def test_should_catch_very_deep_query():
    query = '{\n    user {\n      pets {\n        owner {\n          pets {\n            owner {\n              pets {\n                name\n              }\n            }\n          }\n        }\n      }\n    }\n    }\n    '
    errors, result = run_query(query, 4)
    assert len(errors) == 1
    assert errors[0].message == "'anonymous' exceeds maximum operation depth of 4."