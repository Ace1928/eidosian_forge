from ..json import JSONString
from ..objecttype import ObjectType
from ..schema import Schema
def test_jsonstring_query():
    json_value = '{"key": "value"}'
    json_value_quoted = json_value.replace('"', '\\"')
    result = schema.execute('{ json(input: "%s") }' % json_value_quoted)
    assert not result.errors
    assert result.data == {'json': json_value}
    result = schema.execute('{ json(input: "{}") }')
    assert not result.errors
    assert result.data == {'json': '{}'}