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
def canonical_uri(self, http_request):
    path = urllib.parse.urlparse(http_request.path)
    path_str = six.ensure_str(path.path)
    unquoted = urllib.parse.unquote(path_str)
    encoded = urllib.parse.quote(unquoted, safe='/~')
    return encoded