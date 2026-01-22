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
def test_valid_num_to_float_fromjson(self):
    values = (2, 2.3)
    for v in values:
        for ba in (True, False):
            jd = '%f' if ba else '{"a": %f}'
            i = parse(jd % v, {'a': float}, ba)
            self.assertEqual(i, {'a': float(v)})
            self.assertIsInstance(i['a'], float)