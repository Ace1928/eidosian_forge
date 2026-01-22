import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@fromjson.when_object(bytes)
def str_fromjson(datatype, value):
    if isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
        return str(value).encode('utf8')