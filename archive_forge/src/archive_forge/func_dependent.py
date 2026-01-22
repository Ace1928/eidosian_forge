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
def dependent(field, *groups):

    def decorator(func):

        def wrapper(*args, **kw):
            hasgroup = lambda group: all((key in kw for key in group))
            if field in kw and (not any((hasgroup(g) for g in groups))):
                message = ' OR '.join(['+'.join(g) for g in groups])
                message = '{0} argument {1} requires {2}'.format(func.action, field, message)
                raise KeyError(message)
            return func(*args, **kw)
        message = ' OR '.join(['+'.join(g) for g in groups])
        wrapper.__doc__ = '{0}\n{1} requires: {2}'.format(func.__doc__, field, message)
        return add_attrs_from(func, to=wrapper)
    return decorator