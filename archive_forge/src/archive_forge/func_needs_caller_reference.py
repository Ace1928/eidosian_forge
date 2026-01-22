import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
def needs_caller_reference(func):

    def wrapper(*args, **kw):
        kw.setdefault('CallerReference', uuid.uuid4())
        return func(*args, **kw)
    wrapper.__doc__ = '{0}\nUses CallerReference, defaults to uuid.uuid4()'.format(func.__doc__)
    return add_attrs_from(func, to=wrapper)