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
def iter_call(self, call, *args, **kw):
    """Pass a call name as the first argument and a generator
           is returned for the initial response and any continuation
           call responses made using the NextToken.
        """
    method = self.method_for(call)
    assert method, 'No call named "{0}"'.format(call)
    return self.iter_response(method(*args, **kw))