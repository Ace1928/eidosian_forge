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
def get_proxy_url_with_auth(self):
    if not self.use_proxy:
        return None
    if self.proxy_user or self.proxy_pass:
        if self.proxy_pass:
            login_info = '%s:%s@' % (self.proxy_user, self.proxy_pass)
        else:
            login_info = '%s@' % self.proxy_user
    else:
        login_info = ''
    return 'http://%s%s:%s' % (login_info, self.proxy, str(self.proxy_port or self.port))