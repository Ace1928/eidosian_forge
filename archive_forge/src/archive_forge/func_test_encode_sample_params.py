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
def test_encode_sample_params(self):
    r = wsme.rest.json.encode_sample_params([('a', int, 2)], True)
    assert r[0] == 'javascript', r[0]
    assert r[1] == '{\n    "a": 2\n}', r[1]