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
def test_parse_unexpected_nested_attribute(self):
    no = {'o': {'id': '1', 'name': 'test', 'other': 'unknown'}}
    for ba in (False, True):
        jd = no if ba else {'no': no}
        try:
            parse(json.dumps(jd), {'no': NestedObj}, ba)
        except wsme.exc.UnknownAttribute as e:
            self.assertEqual(e.attributes, set(['other']))
            self.assertEqual(e.fieldname, 'no.o')