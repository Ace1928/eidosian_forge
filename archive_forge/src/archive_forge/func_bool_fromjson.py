import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@fromjson.when_object(bool)
def bool_fromjson(datatype, value):
    """Convert to bool, restricting strings to just unambiguous values."""
    if value is None:
        return None
    if isinstance(value, (int, bool)):
        return bool(value)
    if value in ENUM_TRUE:
        return True
    if value in ENUM_FALSE:
        return False
    raise ValueError('Value not an unambiguous boolean: %s' % value)