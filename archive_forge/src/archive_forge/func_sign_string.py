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
def sign_string(self, string_to_sign):
    new_hmac = self._get_hmac()
    new_hmac.update(string_to_sign.encode('utf-8'))
    return encodebytes(new_hmac.digest()).decode('utf-8').strip()