from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
def get_list(self, action, params, markers, path='/', parent=None, verb='GET'):
    if not parent:
        parent = self
    response = self.make_request(action, params, path, verb)
    body = response.read()
    boto.log.debug(body)
    if not body:
        boto.log.error('Null body %s' % body)
        raise self.ResponseError(response.status, response.reason, body)
    elif response.status == 200:
        rs = ResultSet(markers)
        h = boto.handler.XmlHandler(rs, parent)
        if isinstance(body, six.text_type):
            body = body.encode('utf-8')
        xml.sax.parseString(body, h)
        return rs
    else:
        boto.log.error('%s %s' % (response.status, response.reason))
        boto.log.error('%s' % body)
        raise self.ResponseError(response.status, response.reason, body)