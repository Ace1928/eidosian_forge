from ..json import JSONString
from ..objecttype import ObjectType
from ..schema import Schema
def test_jsonstring_optional_uuid_input():
    """
    Test that we can provide a null value to an optional input
    """
    result = schema.execute('{ json(input: null) }')
    assert not result.errors
    assert result.data == {'json': None}