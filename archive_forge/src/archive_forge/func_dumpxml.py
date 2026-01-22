import base64
import datetime
import decimal
from wsme.rest.xml import fromxml, toxml
import wsme.tests.protocol
from wsme.types import isarray, isdict, isusertype, register_type
from wsme.utils import parse_isodatetime, parse_isodate, parse_isotime
def dumpxml(key, obj, datatype=None):
    el = et.Element(key)
    if isinstance(obj, tuple):
        obj, datatype = obj
    if isinstance(datatype, list):
        for item in obj:
            el.append(dumpxml('item', item, datatype[0]))
    elif isinstance(datatype, dict):
        key_type, value_type = list(datatype.items())[0]
        for item in obj.items():
            node = et.SubElement(el, 'item')
            node.append(dumpxml('key', item[0], key_type))
            node.append(dumpxml('value', item[1], value_type))
    elif datatype == wsme.types.binary:
        el.text = base64.encodebytes(obj).decode('ascii')
    elif isinstance(obj, wsme.types.bytes):
        el.text = obj.decode('ascii')
    elif isinstance(obj, wsme.types.text):
        el.text = obj
    elif type(obj) in (int, float, bool, decimal.Decimal):
        el.text = str(obj)
    elif type(obj) in (datetime.date, datetime.time, datetime.datetime):
        el.text = obj.isoformat()
    elif isinstance(obj, type(None)):
        el.set('nil', 'true')
    elif hasattr(datatype, '_wsme_attributes'):
        for attr in datatype._wsme_attributes:
            name = attr.name
            if name not in obj:
                continue
            o = obj[name]
            el.append(dumpxml(name, o, attr.datatype))
    elif type(obj) is dict:
        for name, value in obj.items():
            el.append(dumpxml(name, value))
    print(obj, datatype, et.tostring(el))
    return el