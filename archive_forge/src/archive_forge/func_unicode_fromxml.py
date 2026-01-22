import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@fromxml.when_object(wsme.types.text)
def unicode_fromxml(datatype, element):
    if element.get('nil') == 'true':
        return None
    return wsme.types.text(element.text) if element.text else ''