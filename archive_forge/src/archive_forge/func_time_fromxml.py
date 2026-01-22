import datetime
import xml.etree.ElementTree as et
from simplegeneric import generic
import wsme.types
from wsme.exc import UnknownArgument, InvalidInput
import re
@fromxml.when_object(datetime.time)
def time_fromxml(datatype, element):
    if element.get('nil') == 'true':
        return None
    return wsme.utils.parse_isotime(element.text)