from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
def destructure_object(value, into, prefix, members=False):
    if isinstance(value, boto.mws.response.ResponseElement):
        destructure_object(value.__dict__, into, prefix, members=members)
    elif isinstance(value, abc.Mapping):
        for name in value:
            if name.startswith('_'):
                continue
            destructure_object(value[name], into, prefix + '.' + name, members=members)
    elif isinstance(value, six.string_types):
        into[prefix] = value
    elif isinstance(value, abc.Iterable):
        for index, element in enumerate(value):
            suffix = (members and '.member.' or '.') + str(index + 1)
            destructure_object(element, into, prefix + suffix, members=members)
    elif isinstance(value, bool):
        into[prefix] = str(value).lower()
    else:
        into[prefix] = value