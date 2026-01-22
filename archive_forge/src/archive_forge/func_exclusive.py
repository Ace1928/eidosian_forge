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
def exclusive(*groups):

    def decorator(func):

        def wrapper(*args, **kw):
            hasgroup = lambda group: all((key in kw for key in group))
            if len(list(filter(hasgroup, groups))) not in (0, 1):
                message = ' OR '.join(['+'.join(g) for g in groups])
                message = '{0} requires either {1}'.format(func.action, message)
                raise KeyError(message)
            return func(*args, **kw)
        message = ' OR '.join(['+'.join(g) for g in groups])
        wrapper.__doc__ = '{0}\nEither: {1}'.format(func.__doc__, message)
        return add_attrs_from(func, to=wrapper)
    return decorator