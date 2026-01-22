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
def detect_potential_sigv4(func):

    def _wrapper(self):
        if os.environ.get('EC2_USE_SIGV4', False):
            return ['hmac-v4']
        if boto.config.get('ec2', 'use-sigv4', False):
            return ['hmac-v4']
        if hasattr(self, 'region'):
            if getattr(self.region, 'endpoint', ''):
                for test in SIGV4_DETECT:
                    if test in self.region.endpoint:
                        return ['hmac-v4']
        return func(self)
    return _wrapper