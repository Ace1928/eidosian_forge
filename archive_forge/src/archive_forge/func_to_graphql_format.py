import typing
from re import sub
def to_graphql_format(data: typing.Dict) -> typing.Dict:
    """
    converts all keys in the data to
    camelcase
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[to_camel_case(key)] = to_graphql_format(value)
        elif isinstance(value, list):
            values = []
            for item in value:
                if isinstance(item, (dict, list)):
                    values.append(to_graphql_format(item))
                else:
                    values.append(item)
            result[to_camel_case(key)] = values
        else:
            result[to_camel_case(key)] = value
    return result