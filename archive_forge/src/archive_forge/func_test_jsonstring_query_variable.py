from ..json import JSONString
from ..objecttype import ObjectType
from ..schema import Schema
def test_jsonstring_query_variable():
    json_value = '{"key": "value"}'
    result = schema.execute('query Test($json: JSONString){ json(input: $json) }', variables={'json': json_value})
    assert not result.errors
    assert result.data == {'json': json_value}