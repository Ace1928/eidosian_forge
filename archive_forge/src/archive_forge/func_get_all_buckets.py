import xml.sax
import base64
import time
from boto.compat import six, urllib
from boto.auth import detect_potential_s3sigv4
import boto.utils
from boto.connection import AWSAuthConnection
from boto import handler
from boto.s3.bucket import Bucket
from boto.s3.key import Key
from boto.resultset import ResultSet
from boto.exception import BotoClientError, S3ResponseError
from boto.utils import get_utf8able_str
def get_all_buckets(self, headers=None):
    response = self.make_request('GET', headers=headers)
    body = response.read()
    if response.status > 300:
        raise self.provider.storage_response_error(response.status, response.reason, body)
    rs = ResultSet([('Bucket', self.bucket_class)])
    h = handler.XmlHandler(rs, self)
    if not isinstance(body, bytes):
        body = body.encode('utf-8')
    xml.sax.parseString(body, h)
    return rs