import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
def encode_error(context, errordetail):
    return json.dumps(errordetail)