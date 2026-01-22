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
def test_form_urlencoded_args(self):
    params = {'value[0].inner.aint': 54, 'value[1].inner.aint': 55}
    body = urlencode(params)
    r = self.app.post('/argtypes/setnestedarray.json', body, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    print(r)
    assert json.loads(r.text) == [{'inner': {'aint': 54}}, {'inner': {'aint': 55}}]