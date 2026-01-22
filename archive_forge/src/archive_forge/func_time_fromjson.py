import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@fromjson.when_object(datetime.time)
def time_fromjson(datatype, value):
    if value is None:
        return None
    return wsme.utils.parse_isotime(value)