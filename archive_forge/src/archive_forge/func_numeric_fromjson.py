import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@fromjson.when_object(int, float)
def numeric_fromjson(datatype, value):
    """Convert string object to built-in types int, long or float."""
    if value is None:
        return None
    return datatype(value)