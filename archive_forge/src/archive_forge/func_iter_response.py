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
def iter_response(self, response):
    """Pass a call's response as the initial argument and a
           generator is returned for the initial response and any
           continuation call responses made using the NextToken.
        """
    yield response
    more = self.method_for(response._action + 'ByNextToken')
    while more and response._result.HasNext == 'true':
        response = more(NextToken=response._result.NextToken)
        yield response