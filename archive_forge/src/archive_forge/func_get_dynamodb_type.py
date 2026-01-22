import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def get_dynamodb_type(val, use_boolean=True):
    """
    Take a scalar Python value and return a string representing
    the corresponding Amazon DynamoDB type.  If the value passed in is
    not a supported type, raise a TypeError.
    """
    dynamodb_type = None
    if val is None:
        dynamodb_type = 'NULL'
    elif is_num(val):
        if isinstance(val, bool) and use_boolean:
            dynamodb_type = 'BOOL'
        else:
            dynamodb_type = 'N'
    elif is_str(val):
        dynamodb_type = 'S'
    elif isinstance(val, (set, frozenset)):
        if False not in map(is_num, val):
            dynamodb_type = 'NS'
        elif False not in map(is_str, val):
            dynamodb_type = 'SS'
        elif False not in map(is_binary, val):
            dynamodb_type = 'BS'
    elif is_binary(val):
        dynamodb_type = 'B'
    elif isinstance(val, Mapping):
        dynamodb_type = 'M'
    elif isinstance(val, list):
        dynamodb_type = 'L'
    if dynamodb_type is None:
        msg = 'Unsupported type "%s" for value "%s"' % (type(val), val)
        raise TypeError(msg)
    return dynamodb_type