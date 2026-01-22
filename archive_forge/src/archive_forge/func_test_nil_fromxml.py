import base64
import datetime
import decimal
from wsme.rest.xml import fromxml, toxml
import wsme.tests.protocol
from wsme.types import isarray, isdict, isusertype, register_type
from wsme.utils import parse_isodatetime, parse_isodate, parse_isotime
def test_nil_fromxml(self):
    for dt in (str, [int], {int: str}, bool, datetime.date, datetime.time, datetime.datetime):
        e = et.Element('value', nil='true')
        assert fromxml(dt, e) is None