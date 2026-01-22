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
def test_true_strings_to_bool(self):
    true_values = ('true', 't', 'yes', 'y', 'on', '1')
    for value in true_values:
        for ba in (True, False):
            jd = '"%s"' if ba else '{"a": "%s"}'
            i = parse(jd % value, {'a': bool}, ba)
            self.assertIsInstance(i['a'], bool)
            self.assertTrue(i['a'])