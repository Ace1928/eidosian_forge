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
def structured_lists(*fields):

    def decorator(func):

        def wrapper(self, *args, **kw):
            for key, acc in [f.split('.') for f in fields]:
                if key in kw:
                    newkey = key + '.' + acc + (acc and '.' or '')
                    for i in range(len(kw[key])):
                        kw[newkey + str(i + 1)] = kw[key][i]
                    kw.pop(key)
            return func(self, *args, **kw)
        wrapper.__doc__ = '{0}\nLists: {1}'.format(func.__doc__, ', '.join(fields))
        return add_attrs_from(func, to=wrapper)
    return decorator