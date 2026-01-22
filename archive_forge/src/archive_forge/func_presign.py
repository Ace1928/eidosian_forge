import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
def presign(self, req, expires, iso_date=None):
    """
        Presign a request using SigV4 query params. Takes in an HTTP request
        and an expiration time in seconds and returns a URL.

        http://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-query-string-auth.html
        """
    if iso_date is None:
        iso_date = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    region = self.determine_region_name(req.host)
    service = self.determine_service_name(req.host)
    params = {'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-Credential': '%s/%s/%s/%s/aws4_request' % (self._provider.access_key, iso_date[:8], region, service), 'X-Amz-Date': iso_date, 'X-Amz-Expires': expires, 'X-Amz-SignedHeaders': 'host'}
    if self._provider.security_token:
        params['X-Amz-Security-Token'] = self._provider.security_token
    headers_to_sign = self.headers_to_sign(req)
    l = sorted(['%s' % n.lower().strip() for n in headers_to_sign])
    params['X-Amz-SignedHeaders'] = ';'.join(l)
    req.params.update(params)
    cr = self.canonical_request(req)
    cr = '\n'.join(cr.split('\n')[:-1]) + '\nUNSIGNED-PAYLOAD'
    req.headers['X-Amz-Date'] = iso_date
    sts = self.string_to_sign(req, cr)
    signature = self.signature(req, sts)
    req.params['X-Amz-Signature'] = signature
    return '%s://%s%s?%s' % (req.protocol, req.host, req.path, urllib.parse.urlencode(req.params))