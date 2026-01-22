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
def string_to_sign(self, http_request, canonical_request):
    """
        Return the canonical StringToSign as well as a dict
        containing the original version of all headers that
        were included in the StringToSign.
        """
    sts = ['AWS4-HMAC-SHA256']
    sts.append(http_request.headers['X-Amz-Date'])
    sts.append(self.credential_scope(http_request))
    sts.append(sha256(canonical_request.encode('utf-8')).hexdigest())
    return '\n'.join(sts)