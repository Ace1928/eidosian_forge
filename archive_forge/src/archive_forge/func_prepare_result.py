import base64
import datetime
import decimal
import json
from urllib.parse import urlencode
from wsme.exc import ClientSideError, InvalidInput
from wsme.types import isarray, isdict, isusertype, register_type
from wsme.types import UserType, ArrayType, DictType
from wsme.rest import expose, validate
from wsme.rest.json import fromjson, tojson, parse
import wsme.tests.protocol
from wsme.utils import parse_isodatetime, parse_isotime, parse_isodate
def prepare_result(value, datatype):
    print(value, datatype)
    if value is None:
        return None
    if datatype == wsme.types.binary:
        return base64.decodebytes(value.encode('ascii'))
    if isusertype(datatype):
        datatype = datatype.basetype
    if isinstance(datatype, list):
        return [prepare_result(item, datatype[0]) for item in value]
    if isarray(datatype):
        return [prepare_result(item, datatype.item_type) for item in value]
    if isinstance(datatype, dict):
        return dict(((prepare_result(item[0], list(datatype.keys())[0]), prepare_result(item[1], list(datatype.values())[0])) for item in value.items()))
    if isdict(datatype):
        return dict(((prepare_result(item[0], datatype.key_type), prepare_result(item[1], datatype.value_type)) for item in value.items()))
    if datatype == datetime.date:
        return parse_isodate(value)
    if datatype == datetime.time:
        return parse_isotime(value)
    if datatype == datetime.datetime:
        return parse_isodatetime(value)
    if hasattr(datatype, '_wsme_attributes'):
        for attr in datatype._wsme_attributes:
            if attr.key not in value:
                continue
            value[attr.key] = prepare_result(value[attr.key], attr.datatype)
        return value
    if datatype == wsme.types.bytes:
        return value.encode('ascii')
    if type(value) is not datatype:
        print(type(value), datatype)
        return datatype(value)
    return value