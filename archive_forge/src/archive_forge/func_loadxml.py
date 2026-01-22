import base64
import datetime
import decimal
from wsme.rest.xml import fromxml, toxml
import wsme.tests.protocol
from wsme.types import isarray, isdict, isusertype, register_type
from wsme.utils import parse_isodatetime, parse_isodate, parse_isotime
def loadxml(el, datatype):
    print(el, datatype, len(el))
    if el.get('nil') == 'true':
        return None
    if isinstance(datatype, list):
        return [loadxml(item, datatype[0]) for item in el.findall('item')]
    elif isarray(datatype):
        return [loadxml(item, datatype.item_type) for item in el.findall('item')]
    elif isinstance(datatype, dict):
        key_type, value_type = list(datatype.items())[0]
        return dict(((loadxml(item.find('key'), key_type), loadxml(item.find('value'), value_type)) for item in el.findall('item')))
    elif isdict(datatype):
        return dict(((loadxml(item.find('key'), datatype.key_type), loadxml(item.find('value'), datatype.value_type)) for item in el.findall('item')))
    elif isdict(datatype):
        return dict(((loadxml(item.find('key'), datatype.key_type), loadxml(item.find('value'), datatype.value_type)) for item in el.findall('item')))
    elif len(el):
        d = {}
        for attr in datatype._wsme_attributes:
            name = attr.name
            child = el.find(name)
            print(name, attr, child)
            if child is not None:
                d[name] = loadxml(child, attr.datatype)
        print(d)
        return d
    else:
        if datatype == wsme.types.binary:
            return base64.decodebytes(el.text.encode('ascii'))
        if isusertype(datatype):
            datatype = datatype.basetype
        if datatype == datetime.date:
            return parse_isodate(el.text)
        if datatype == datetime.time:
            return parse_isotime(el.text)
        if datatype == datetime.datetime:
            return parse_isodatetime(el.text)
        if datatype == wsme.types.text:
            return datatype(el.text if el.text else '')
        if datatype == bool:
            return el.text.lower() != 'false'
        if datatype is None:
            return el.text
        if datatype is wsme.types.bytes:
            return el.text.encode('ascii')
        return datatype(el.text)